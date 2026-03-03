import json, asyncio, logging, time
from langchain_core.messages import SystemMessage, HumanMessage
from app.llm import get_llm
from app.models import DocPage, RefineRouter, MetaUpdate

log = logging.getLogger("agent")

ROUTER_PROMPT = """You are a documentation routing agent. Given a table of contents with page summaries and a user's refinement request, identify which pages need to be modified.

- page_ids: array of page IDs that need content changes
- include_meta: true if the overall title/description should change
- strategy: one sentence describing the refinement approach

If the request is vague or applies broadly (e.g. "make it better"), include all page IDs."""

REFINE_PAGE_PROMPT = """You are a documentation writer agent for "better-docs". You receive ONE documentation page and a refinement instruction.

Apply the refinement and return the COMPLETE updated page. Preserve sections the instruction doesn't mention. No emojis. Professional, clean language. Include real code examples where relevant."""

REFINE_META_PROMPT = """You are a documentation metadata agent. Update the title and description based on the instruction."""


async def refine_docs(current_docs: dict, prompt: str, repo_name: str) -> dict:
    t0 = time.time()
    llm = get_llm()
    router_llm = llm.with_structured_output(RefineRouter)
    meta_llm = llm.with_structured_output(MetaUpdate)
    page_writer_llm = llm.with_structured_output(DocPage)

    pages = current_docs.get("pages", {})
    navigation = current_docs.get("navigation", [])

    toc_lines = []
    for group in navigation:
        toc_lines.append(f"## {group.get('group', '')}")
        for page_id in group.get("pages", []):
            page = pages.get(page_id, {})
            desc = page.get("description", "")
            n_sections = len(page.get("sections", []))
            toc_lines.append(f"  - {page_id}: {page.get('title', page_id)} ({n_sections} sections) -- {desc}")
    toc = "\n".join(toc_lines)

    # --- Agent 1: Router ---
    log.info("[refine/router] Identifying target pages for: %s", prompt[:80])
    router_msg = f"""Documentation for "{repo_name}":

{toc}

User request: {prompt}"""

    try:
        plan: RefineRouter = await asyncio.wait_for(
            router_llm.ainvoke([SystemMessage(content=ROUTER_PROMPT), HumanMessage(content=router_msg)]),
            timeout=30,
        )
        target_ids = plan.page_ids
        include_meta = plan.include_meta
        strategy = plan.strategy
        log.info("[refine/router] Strategy: %s | Targets: %s | Meta: %s", strategy, target_ids, include_meta)
    except Exception as e:
        log.warning("[refine/router] Failed, refining all pages: %s", e)
        target_ids = list(pages.keys())
        include_meta = False
        strategy = prompt

    updated_docs = {**current_docs, "pages": {**pages}}

    # --- Agent 2: Meta agent ---
    async def _refine_meta() -> None:
        if not include_meta:
            return
        try:
            meta_msg = f"""Current title: {current_docs.get('title', '')}
Current description: {current_docs.get('description', '')}

Instruction: {strategy}"""
            meta: MetaUpdate = await asyncio.wait_for(
                meta_llm.ainvoke([SystemMessage(content=REFINE_META_PROMPT), HumanMessage(content=meta_msg)]),
                timeout=30,
            )
            updated_docs["title"] = meta.title
            updated_docs["description"] = meta.description
            log.info("[refine/meta] Updated title/description")
        except Exception as e:
            log.error("[refine/meta] Failed: %s", e)

    # --- Agent 3 (x N): Page writer agents ---
    async def _refine_one_page(page_id: str) -> tuple[str, dict]:
        """Refine a single page. On failure, returns the original page unchanged."""
        page_data = pages.get(page_id)
        if not page_data:
            return page_id, page_data or {}

        page_json = json.dumps(page_data, indent=2)
        if len(page_json) > 15000:
            page_json = json.dumps(page_data)[:15000]

        user_msg = f"""Page "{page_id}" from "{repo_name}" docs:
{page_json}

Instruction: {strategy}"""

        try:
            result: DocPage = await asyncio.wait_for(
                page_writer_llm.ainvoke([SystemMessage(content=REFINE_PAGE_PROMPT), HumanMessage(content=user_msg)]),
                timeout=90,
            )
            log.info("[refine/writer] Page '%s' refined successfully", page_id)
            return page_id, result.model_dump(exclude_none=True)
        except Exception as e:
            log.error("[refine/writer] Page '%s' failed, keeping original: %s", page_id, e)
            return page_id, page_data

    sem = asyncio.Semaphore(6)

    async def _limited(page_id: str) -> tuple[str, dict]:
        async with sem:
            return await _refine_one_page(page_id)

    # --- Run all agents concurrently ---
    page_ids_to_refine = [pid for pid in target_ids if pid in pages]
    all_tasks: list = [_refine_meta()]
    all_tasks.extend([_limited(pid) for pid in page_ids_to_refine])

    log.info("[refine] Launching %d page agents + meta agent", len(page_ids_to_refine))
    results = await asyncio.gather(*all_tasks, return_exceptions=True)

    for result in results[1:]:
        if isinstance(result, Exception):
            log.error("[refine] Agent failed: %s", result)
            continue
        page_id, page_data = result
        if page_data:
            updated_docs["pages"][page_id] = page_data

    log.info("[refine] Done in %.1fs -- refined %d pages", time.time() - t0, len(page_ids_to_refine))
    return updated_docs
