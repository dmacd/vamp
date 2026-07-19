"""Notebook-only presentation layer for the TinyWorlds playground."""

from __future__ import annotations

from html import escape
from typing import Callable

from apm.data.text.tinyworlds import DataSplit, EntityId, QueryKind
from apm.interactive.tinyworlds import (
    CalibrationTrialArtifact,
    CandidateDiagnostic,
    QueryInspection,
    TinyWorldsDemo,
    TinyWorldsLab,
    _format_atom,
    _format_rule,
    candidate_diagnostic,
    generate_tinyworlds_demo,
    inspect_query,
)


def build_tinyworlds_playground(
    lab: TinyWorldsLab,
    demo: TinyWorldsDemo,
) -> object:
    """Return a tabbed widget covering generation, proofs, scores, and transfer."""
    try:
        import ipywidgets as widgets
        from IPython.display import HTML, clear_output, display
    except ImportError as error:
        raise ImportError(
            "the TinyWorlds playground requires the notebook extra; "
            "install with `pip install -e '.[notebook,lm]'`"
        ) from error

    layout = widgets.Layout(width="100%")
    task_options = tuple(str(task.task_id) for task in demo.bundle.tasks)
    kind_options = tuple((kind.value, kind.value) for kind in QueryKind)

    world_task = widgets.Dropdown(
        options=task_options,
        description="Task",
        layout=layout,
    )
    world_output = widgets.Output(layout=layout)

    def show_world(task_id: str) -> None:
        with world_output:
            clear_output(wait=True)
            display(HTML(_world_html(demo, task_id)))

    world_task.observe(lambda change: show_world(str(change["new"])), names="value")
    show_world(str(world_task.value))
    world_panel = widgets.VBox((world_task, world_output), layout=layout)

    story_task = widgets.Dropdown(
        options=task_options,
        description="Task",
        layout=layout,
    )
    story_split = widgets.ToggleButtons(
        options=(("training", "train"), ("validation", "validation")),
        description="Split",
    )
    story_index = widgets.IntSlider(
        value=0,
        min=0,
        max=DEMO_STORY_MAXIMUMS["train"],
        description="Story",
        continuous_update=False,
        layout=layout,
    )
    story_output = widgets.Output(layout=layout)

    def refresh_story(*_changes: object) -> None:
        split = str(story_split.value)
        story_index.max = DEMO_STORY_MAXIMUMS[split]
        story_index.value = min(story_index.value, story_index.max)
        with story_output:
            clear_output(wait=True)
            display(
                HTML(
                    _story_html(
                        demo,
                        str(story_task.value),
                        DataSplit(split),
                        story_index.value,
                    )
                )
            )

    for control in (story_task, story_split, story_index):
        control.observe(refresh_story, names="value")
    refresh_story()
    story_panel = widgets.VBox(
        (
            widgets.HBox((story_task, story_split), layout=layout),
            story_index,
            story_output,
        ),
        layout=layout,
    )

    proof_kind = widgets.Dropdown(
        options=kind_options,
        value=QueryKind.ONE_HOP.value,
        description="Query",
        layout=layout,
    )
    proof_depth = widgets.SelectionSlider(
        options=(0, 1, 2),
        value=1,
        description="Max depth",
        continuous_update=False,
        layout=layout,
    )
    fact_budget = widgets.SelectionSlider(
        options=(0, 12, 24, 36),
        value=36,
        description="Fact budget",
        continuous_update=False,
        layout=layout,
    )
    removed_fact = widgets.Dropdown(
        options=(("none", ""),),
        description="Ablate",
        layout=layout,
    )
    proof_output = widgets.Output(layout=layout)

    def refresh_proof(*_changes: object) -> None:
        inspection = inspect_query(demo, QueryKind(str(proof_kind.value)))
        options = (("none", ""),) + tuple(
            (f"{fact.task_id} #{fact.exposure_position}", fact.fact_id)
            for fact in inspection.support_facts
        )
        if tuple(removed_fact.options) != options:
            removed_fact.options = options
            removed_fact.value = ""
        with proof_output:
            clear_output(wait=True)
            display(
                HTML(
                    _query_html(
                        inspection,
                        max_depth=int(proof_depth.value),
                        fact_budget=int(fact_budget.value),
                        removed_fact_id=str(removed_fact.value),
                    )
                )
            )

    for control in (proof_kind, proof_depth, fact_budget, removed_fact):
        control.observe(refresh_proof, names="value")
    refresh_proof()
    proof_panel = widgets.VBox(
        (
            proof_kind,
            widgets.HBox((proof_depth, fact_budget), layout=layout),
            removed_fact,
            proof_output,
        ),
        layout=layout,
    )

    rendering_kind = widgets.Dropdown(
        options=kind_options,
        value=QueryKind.DIRECT.value,
        description="Query",
        layout=layout,
    )
    prefix_length = widgets.ToggleButtons(
        options=(64, 128, 192),
        value=64,
        description="Prefix",
    )
    rendering_output = widgets.Output(layout=layout)

    def refresh_rendering(*_changes: object) -> None:
        with rendering_output:
            clear_output(wait=True)
            display(
                HTML(
                    _rendering_html(
                        demo,
                        QueryKind(str(rendering_kind.value)),
                        int(prefix_length.value),
                    )
                )
            )

    for control in (rendering_kind, prefix_length):
        control.observe(refresh_rendering, names="value")
    refresh_rendering()
    rendering_panel = widgets.VBox(
        (widgets.HBox((rendering_kind, prefix_length)), rendering_output),
        layout=layout,
    )

    address_kind = widgets.Dropdown(
        options=kind_options,
        value=QueryKind.CROSS_BRANCH.value,
        description="Query",
        layout=layout,
    )
    address_node = widgets.Dropdown(
        options=("root", *task_options),
        value="root",
        description="Hard node",
        layout=layout,
    )
    coefficient_sliders = tuple(
        (
            task_id,
            widgets.FloatSlider(
                value=0.0,
                min=0.0,
                max=1.0,
                step=0.05,
                description=task_id,
                continuous_update=False,
                readout_format=".2f",
                layout=layout,
            ),
        )
        for task_id in task_options
    )
    load_hard_path = widgets.Button(
        description="Copy hard path to sliders",
        icon="copy",
    )
    address_output = widgets.Output(layout=layout)

    def refresh_address(*_changes: object) -> None:
        inspection = inspect_query(demo, QueryKind(str(address_kind.value)))
        coefficients = {
            f"edge:{task_id}": float(slider.value)
            for task_id, slider in coefficient_sliders
        }
        with address_output:
            clear_output(wait=True)
            display(
                HTML(
                    _addressing_html(
                        inspection,
                        str(address_node.value),
                        coefficients,
                    )
                )
            )

    def copy_hard_path(_button: object) -> None:
        selected = str(address_node.value)
        path = (
            ()
            if selected == "root"
            else tuple(
                str(value)
                for value in demo.bundle.world.task_path(
                    next(
                        task.task_id
                        for task in demo.bundle.tasks
                        if str(task.task_id) == selected
                    )
                )
            )
        )
        for task_id, slider in coefficient_sliders:
            slider.value = float(task_id in path)
        refresh_address()

    for control in (address_kind, address_node, *(item[1] for item in coefficient_sliders)):
        control.observe(refresh_address, names="value")
    load_hard_path.on_click(copy_hard_path)
    refresh_address()
    address_panel = widgets.VBox(
        (
            widgets.HBox((address_kind, address_node)),
            load_hard_path,
            widgets.Accordion(
                children=(
                    widgets.VBox(tuple(slider for _, slider in coefficient_sliders)),
                ),
                titles=("Continuous edge coefficients",),
            ),
            address_output,
        ),
        layout=layout,
    )

    facts_control = widgets.ToggleButtons(
        options=(("24 facts", 24), ("12 facts", 12), ("36 facts", 36)),
        value=24,
        description="Trial",
    )
    score_metrics = tuple(
        dict.fromkeys(row.metric for row in lab.trials[0].candidate_scores)
    )
    metric_control = widgets.Dropdown(
        options=score_metrics,
        value="frozen_novel_binding",
        description="Metric",
        layout=layout,
    )
    score_output = widgets.Output(layout=layout)

    def refresh_score(*_changes: object) -> None:
        with score_output:
            clear_output(wait=True)
            try:
                diagnostic = candidate_diagnostic(
                    lab,
                    demo,
                    facts_per_task=int(facts_control.value),
                    metric=str(metric_control.value),
                )
            except (KeyError, ValueError) as error:
                display(
                    HTML(
                        "<p><strong>Saved NLLs are bound to the seed-0 hard "
                        "calibration demo.</strong></p>"
                        f"<p>{escape(str(error))}</p>"
                    )
                )
                return
            display(HTML(_candidate_html(diagnostic)))
            figure = _candidate_figure(diagnostic)
            display(figure)
            _close_figure(figure)

    for control in (facts_control, metric_control):
        control.observe(refresh_score, names="value")
    refresh_score()
    score_panel = widgets.VBox(
        (widgets.HBox((facts_control, metric_control)), score_output),
        layout=layout,
    )

    gate_output = widgets.Output(layout=layout)
    with gate_output:
        display(HTML(_gate_html(lab)))
        gate_figure = _calibration_figure(lab)
        display(gate_figure)
        _close_figure(gate_figure)
    gate_panel = widgets.VBox((gate_output,), layout=layout)

    transfer_facts = widgets.ToggleButtons(
        options=(("24 facts", 24), ("12 facts", 12), ("36 facts", 36)),
        value=24,
        description="Trial",
    )
    transfer_task = widgets.Dropdown(
        options=task_options,
        value=task_options[2],
        description="Task",
        layout=layout,
    )
    transfer_output = widgets.Output(layout=layout)

    def refresh_transfer(*_changes: object) -> None:
        artifact = lab.trial_for_facts(int(transfer_facts.value))
        with transfer_output:
            clear_output(wait=True)
            display(HTML(_topology_and_parent_html(demo, artifact)))
            figure = _transfer_figure(artifact, str(transfer_task.value))
            display(figure)
            _close_figure(figure)

    for control in (transfer_facts, transfer_task):
        control.observe(refresh_transfer, names="value")
    refresh_transfer()
    transfer_panel = widgets.VBox(
        (widgets.HBox((transfer_facts, transfer_task)), transfer_output),
        layout=layout,
    )

    generation_seed = widgets.BoundedIntText(
        value=1,
        min=0,
        max=2**31 - 1,
        description="Public seed",
    )
    generation_world = widgets.Dropdown(
        options=("calibration", "pilot"),
        value="calibration",
        description="World",
    )
    generation_facts = widgets.ToggleButtons(
        options=(24, 36),
        value=24,
        description="Fact capacity",
    )
    generation_policy = widgets.Dropdown(
        options=("hard", "standard_mix"),
        value="hard",
        description="Distractors",
    )
    generation_button = widgets.Button(
        description="Generate sample",
        button_style="primary",
        icon="refresh",
    )
    generation_output = widgets.Output(layout=layout)

    def generate_sample(_button: object) -> None:
        generation_button.disabled = True
        with generation_output:
            clear_output(wait=True)
            display(HTML("<p><strong>Rendering through the production templates…</strong></p>"))
        try:
            sample = generate_tinyworlds_demo(
                lab,
                public_seed=int(generation_seed.value),
                world_name=str(generation_world.value),
                fact_capacity=int(generation_facts.value),
                distractor_policy=str(generation_policy.value),
            )
            first_task = str(sample.bundle.tasks[0].task_id)
            with generation_output:
                clear_output(wait=True)
                display(HTML(_world_html(sample, first_task)))
                display(
                    HTML(
                        _story_html(
                            sample,
                            first_task,
                            DataSplit.TRAIN,
                            0,
                        )
                    )
                )
        finally:
            generation_button.disabled = False

    generation_button.on_click(generate_sample)
    generation_panel = widgets.VBox(
        (
            widgets.HBox((generation_seed, generation_world)),
            widgets.HBox((generation_facts, generation_policy)),
            generation_button,
            generation_output,
        ),
        layout=layout,
    )

    tabs = widgets.Tab(
        children=(
            world_panel,
            story_panel,
            proof_panel,
            rendering_panel,
            address_panel,
            score_panel,
            gate_panel,
            transfer_panel,
            generation_panel,
        ),
        layout=layout,
    )
    for index, title in enumerate(
        (
            "World",
            "Stories",
            "Proofs",
            "Cues",
            "Addressing",
            "Candidate NLL",
            "Gates",
            "Transfer",
            "Generate",
        )
    ):
        tabs.set_title(index, title)
    return tabs


DEMO_STORY_MAXIMUMS = {
    "train": 11,
    "validation": 2,
}


def _world_html(demo: TinyWorldsDemo, task_id: str) -> str:
    task = next(value for value in demo.bundle.tasks if str(value.task_id) == task_id)
    names = {entity.entity_id: entity.name for entity in demo.bundle.entities}
    facts_by_id = {fact.atom_id: fact for fact in demo.bundle.facts}
    rules_by_id = {rule.rule_id: rule for rule in demo.bundle.rules}
    entity_rows = "".join(
        f"<li><code>{escape(names[entity_id])}</code> "
        f"<small>({escape(str(entity_id))})</small></li>"
        for entity_id in task.introduced_entity_ids
    )
    fact_rows = "".join(
        f"<li><code>{escape(_format_atom(facts_by_id[fact_id], names))}</code></li>"
        for fact_id in task.direct_fact_ids[:8]
    )
    rule_rows = "".join(
        f"<li><code>{escape(_format_rule(rules_by_id[rule_id], names))}</code></li>"
        for rule_id in task.rule_ids
    ) or "<li>none introduced at this task</li>"
    parent = "root" if task.parent_task_id is None else str(task.parent_task_id)
    path = " → ".join(str(value) for value in demo.bundle.world.task_path(task.task_id))
    return (
        f"<h3>{escape(task_id)}</h3>"
        "<dl>"
        f"<dt>kind</dt><dd>{escape(task.kind.value)}</dd>"
        f"<dt>symbolic parent</dt><dd>{escape(parent)}</dd>"
        f"<dt>hard path</dt><dd>{escape(path)}</dd>"
        f"<dt>direct facts</dt><dd>{len(task.direct_fact_ids)}</dd>"
        "</dl>"
        "<details><summary>Introduced entities</summary><ul>"
        f"{entity_rows}</ul></details>"
        "<details open><summary>First eight direct facts</summary><ol>"
        f"{fact_rows}</ol></details>"
        f"<details><summary>Rules</summary><ul>{rule_rows}</ul></details>"
    )


def _story_html(
    demo: TinyWorldsDemo,
    task_id: str,
    split: DataSplit,
    index: int,
) -> str:
    stories = tuple(
        story
        for story in demo.rendered.stories
        if story.task_id == task_id and story.split is split
    )
    story = stories[index % len(stories)]
    alignments = "".join(
        "<tr>"
        f"<td>{alignment.sentence_index}</td>"
        f"<td>{escape(story.text[alignment.start_character:alignment.end_character])}</td>"
        f"<td><code>{escape(', '.join(alignment.fact_ids) or '—')}</code></td>"
        f"<td><code>{escape(', '.join(alignment.rule_ids) or '—')}</code></td>"
        "</tr>"
        for alignment in story.alignments
    )
    return (
        f"<h3>{escape(story.story_id)}</h3>"
        f"<p><strong>{len(story.token_ids)} tokenizer tokens</strong> · "
        f"plot <code>{escape(story.plot_id)}</code></p>"
        f"<blockquote style='white-space:pre-wrap'>{escape(story.text)}</blockquote>"
        "<table><thead><tr><th>#</th><th>exact aligned sentence</th>"
        f"<th>fact IDs</th><th>rule IDs</th></tr></thead><tbody>{alignments}</tbody></table>"
    )


def _query_html(
    inspection: QueryInspection,
    *,
    max_depth: int,
    fact_budget: int,
    removed_fact_id: str,
) -> str:
    answers = dict(inspection.answers_by_max_depth)[max_depth]
    depth_has_answer = inspection.answer_entity_id in answers
    budget_has_support = all(
        fact.exposure_position <= fact_budget for fact in inspection.support_facts
    )
    removed = next(
        (fact for fact in inspection.support_facts if fact.fact_id == removed_fact_id),
        None,
    )
    ablation_text = (
        "No support fact removed."
        if removed is None
        else (
            f"Removing {removed.task_id} fact #{removed.exposure_position} "
            + ("still leaves an answer." if removed.answer_survives_removal else "removes the answer.")
        )
    )
    candidates = "".join(
        "<tr>"
        f"<td>{candidate.index}</td><td><code>{escape(candidate.name)}</code></td>"
        f"<td>{escape(candidate.role)}</td>"
        f"<td>{'✓' if candidate.correct else ''}</td></tr>"
        for candidate in inspection.candidates
    )
    proof = "".join(
        "<tr>"
        f"<td>{step.depth}</td><td><code>{escape(step.atom_text)}</code></td>"
        f"<td><code>{escape(step.rule_id or 'direct fact')}</code></td>"
        f"<td><code>{escape(', '.join(step.premise_atom_ids) or '—')}</code></td>"
        "</tr>"
        for step in inspection.proof_steps
    )
    support = "".join(
        "<tr>"
        f"<td>{escape(fact.task_id)}</td><td>{fact.exposure_position}</td>"
        f"<td><code>{escape(fact.atom_text)}</code></td>"
        f"<td>{'yes' if fact.answer_survives_removal else 'no'}</td></tr>"
        for fact in inspection.support_facts
    )
    hard_support = "".join(
        "<tr>"
        f"<td>{escape(row.node_id)}</td>"
        f"<td><code>{escape(', '.join(row.path_edge_ids))}</code></td>"
        f"<td>{row.required_edge_recall:.0%}</td></tr>"
        for row in inspection.hard_support
    )
    definition = (
        "<p><strong>One-hop means one Horn-rule application.</strong> It does not "
        "mean one transformer layer or one VAMP graph edge.</p>"
        if inspection.kind is QueryKind.ONE_HOP
        else ""
    )
    return (
        f"<h3>{escape(inspection.kind.value)} · proof depth {inspection.proof_depth}</h3>"
        f"{definition}<p><code>{escape(inspection.query_text)}</code></p>"
        f"<p>Answer: <strong>{escape(inspection.answer_name)}</strong></p>"
        "<ul>"
        f"<li>With closure depth ≤ {max_depth}: <strong>{'answer present' if depth_has_answer else 'answer absent'}</strong></li>"
        f"<li>With the first {fact_budget} facts/task: <strong>{'all proof leaves exposed' if budget_has_support else 'proof support incomplete'}</strong></li>"
        f"<li>{escape(ablation_text)}</li>"
        "</ul>"
        "<details><summary>Four candidates</summary><table><thead><tr>"
        f"<th>index</th><th>name</th><th>role</th><th>answer</th></tr></thead><tbody>{candidates}</tbody></table></details>"
        "<details open><summary>Canonical proof</summary><table><thead><tr>"
        f"<th>depth</th><th>atom</th><th>rule</th><th>premises</th></tr></thead><tbody>{proof}</tbody></table></details>"
        "<details open><summary>Support facts and training positions</summary><table><thead><tr>"
        f"<th>task</th><th>position</th><th>fact</th><th>answer survives removal?</th></tr></thead><tbody>{support}</tbody></table></details>"
        "<details><summary>Hard-path required-edge support</summary><table><thead><tr>"
        f"<th>node</th><th>path edges</th><th>recall</th></tr></thead><tbody>{hard_support}</tbody></table></details>"
    )


def _rendering_html(
    demo: TinyWorldsDemo,
    kind: QueryKind,
    prefix_length: int,
) -> str:
    group = next(
        value
        for value in demo.rendered.query_groups
        if value.split is DataSplit.VALIDATION
        and value.group_plan.source_plan.kind is kind
    )
    variant = next(
        value for value in group.variants if len(value.prefix_token_ids) == prefix_length
    )
    query = variant.knowledge_query
    names = {str(entity.entity_id): entity.name for entity in demo.bundle.entities}
    role_by_id = {
        str(candidate.entity_id): candidate.role.value
        for candidate in group.group_plan.candidates
    }
    candidates = "".join(
        "<tr>"
        f"<td>{index}</td><td><code>{escape(names[entity_id])}</code></td>"
        f"<td>{escape(role_by_id[entity_id])}</td>"
        f"<td><code>{escape(candidate.answer_text)}</code></td>"
        f"<td>{int(candidate.competence_batch.loss_mask.sum())}</td></tr>"
        for index, (entity_id, candidate) in enumerate(
            zip(variant.candidate_entity_ids, query.candidates)
        )
    )
    suffix_counts = {
        int(candidate.competence_batch.loss_mask.sum()) for candidate in query.candidates
    }
    return (
        f"<h3>{escape(kind.value)} · {prefix_length} tokens</h3>"
        "<dl>"
        f"<dt>cue regime</dt><dd>{escape(query.cue_regime)}</dd>"
        f"<dt>visible cues</dt><dd><code>{escape(', '.join(query.visible_cue_ids) or 'none')}</code></dd>"
        f"<dt>eligible tasks</dt><dd><code>{escape(', '.join(query.eligible_task_ids))}</code></dd>"
        f"<dt>mode</dt><dd>{escape(query.mode)}</dd>"
        f"<dt>shared final core</dt><dd><code>{escape(variant.query_core_sha256[:16])}…</code></dd>"
        f"<dt>equal active suffix tokens</dt><dd>{len(suffix_counts) == 1} ({next(iter(suffix_counts))})</dd>"
        "</dl>"
        f"<blockquote style='white-space:pre-wrap'>{escape(variant.prefix_text)}</blockquote>"
        "<table><thead><tr><th>index</th><th>name</th><th>role</th>"
        f"<th>exact answer suffix</th><th>active tokens</th></tr></thead><tbody>{candidates}</tbody></table>"
    )


def _candidate_html(diagnostic: CandidateDiagnostic) -> str:
    score = diagnostic.score
    explanations = {
        "frozen_novel_binding": (
            "The frozen model never learned this fictional fact. A correct choice here "
            "reveals candidate-string likelihood, not recall."
        ),
        "independent_direct_recall": (
            "Compare this with frozen direct accuracy: both were 100%, so the measured "
            "training lift was zero."
        ),
        "frozen_one_hop": (
            "This is the unchanged base model on a query requiring one rule application."
        ),
        "independent_one_hop": (
            "Only the 36-fact trial exposed both proof leaves (positions 6 and 31)."
        ),
        "old_contextual_answer": (
            "For this paired revision check, the seed node is deliberately scored "
            "against the old base value, whose candidate role is incompatible_revision."
        ),
        "revision_contextual_answer": (
            "The revision node should choose the new contextual value; it instead "
            "preferred a filler in every measured rendering."
        ),
    }
    explanation = explanations.get(score.metric, "")
    rows = "".join(
        "<tr>"
        f"<td>{candidate.index}</td><td><code>{escape(candidate.name)}</code></td>"
        f"<td>{escape(candidate.role)}</td><td>{candidate.nll:.4f}</td>"
        f"<td>{'✓' if candidate.correct else ''}</td>"
        f"<td>{'← min' if candidate.predicted else ''}</td></tr>"
        for candidate in diagnostic.candidates
    )
    return (
        f"<h3>{escape(score.metric)} · {escape(score.method)}</h3>"
        f"<p>{escape(explanation)}</p>"
        f"<p>Lower NLL wins. Margin = <strong>{score.margin:+.4f}</strong>; "
        f"result = <strong>{'correct' if score.correct else 'wrong'}</strong>.</p>"
        f"<p>Cue: <code>{escape(diagnostic.cue_regime)}</code>; eligible tasks: "
        f"<code>{escape(', '.join(diagnostic.eligible_task_ids))}</code>.</p>"
        "<table><thead><tr><th>index</th><th>nonce name</th><th>role</th>"
        f"<th>NLL</th><th>correct</th><th>predicted</th></tr></thead><tbody>{rows}</tbody></table>"
        "<details><summary>Visible closed-book prefix</summary>"
        f"<blockquote style='white-space:pre-wrap'>{escape(diagnostic.prefix_text)}</blockquote></details>"
    )


def _addressing_html(
    inspection: QueryInspection,
    selected_node_id: str,
    coefficients: dict[str, float],
) -> str:
    required = set(inspection.required_edge_ids)
    soft_support = sum(coefficients.get(edge_id, 0.0) for edge_id in required) / len(
        required
    )
    hard_recall = (
        0.0
        if selected_node_id == "root"
        else next(
            row.required_edge_recall
            for row in inspection.hard_support
            if row.node_id == selected_node_id
        )
    )
    rows = "".join(
        "<tr>"
        f"<td><code>{escape(edge_id)}</code></td>"
        f"<td>{'required' if edge_id in required else ''}</td>"
        f"<td>{coefficient:.2f}</td></tr>"
        for edge_id, coefficient in coefficients.items()
    )
    oracle = ", ".join(inspection.hard_oracle_task_ids) or "none (cross-branch)"
    return (
        f"<h3>{escape(inspection.kind.value)} support algebra</h3>"
        "<p>A hard router selects one root-to-node path. EBT-soft instead supplies "
        "continuous edge coefficients. The sliders show the exact support metrics; "
        "they do not invent an EBT model score that Phase 4 never recorded.</p>"
        "<dl>"
        f"<dt>hard oracle</dt><dd><code>{escape(oracle)}</code></dd>"
        f"<dt>selected hard-node support recall</dt><dd>{hard_recall:.1%}</dd>"
        f"<dt>continuous required-edge support</dt><dd>{soft_support:.3f}</dd>"
        "</dl>"
        "<p>Copying a hard path to the sliders demonstrates the one-hot soft/hard "
        "equivalence. A cross-branch proof cannot reach 100% through any one hard path.</p>"
        "<table><thead><tr><th>edge</th><th>query support</th><th>coefficient</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def _gate_html(lab: TinyWorldsLab) -> str:
    headers = "".join(
        f"<th>{artifact.trial.request.config.facts_per_task} facts</th>"
        for artifact in lab.trials
    )
    gates = lab.trials[0].trial.gate_decisions
    rows = "".join(
        "<tr>"
        f"<td><code>{escape(gate.gate.value)}</code><br><small>{escape(gate.criterion)}</small></td>"
        + "".join(
            f"<td style='color:{'#047857' if decision.passed else '#b91c1c'}'>"
            f"{'PASS' if decision.passed else 'FAIL'}<br>"
            f"<small>{escape(', '.join(f'{value:.3g}' for value in decision.observed))}</small></td>"
            for artifact in lab.trials
            for decision in (artifact.trial.gate_decisions[gate_index],)
        )
        + "</tr>"
        for gate_index, gate in enumerate(gates)
    )
    return (
        "<h3>Phase 4 validation gates</h3>"
        "<p>Every configuration had to pass every row. The 36-fact trial repaired "
        "the one-hop slice, but frozen direct accuracy, direct lift, and revision "
        "consistency still failed, so the ladder stopped.</p>"
        f"<table><thead><tr><th>gate</th>{headers}</tr></thead><tbody>{rows}</tbody></table>"
    )


def _topology_and_parent_html(
    demo: TinyWorldsDemo,
    artifact: CalibrationTrialArtifact,
) -> str:
    expected = tuple(
        (
            "root" if task.parent_task_id is None else str(task.parent_task_id),
            str(task.task_id),
        )
        for task in demo.bundle.tasks
    )
    learned = tuple(
        (str(node.parent_id), node.node_id)
        for node in artifact.learned_graph
        if node.parent_id is not None
    )
    graph_rows = "".join(
        "<tr>"
        f"<td><code>{escape(parent)} → {escape(child)}</code></td>"
        f"<td><code>{escape(l_parent)} → {escape(l_child)}</code></td></tr>"
        for (parent, child), (l_parent, l_child) in zip(expected, learned)
    )
    parent_rows = "".join(
        "<tr>"
        f"<td>{escape(row.task_id)}</td><td>{escape(row.selected_node_id)}</td>"
        f"<td>{escape(', '.join(f'{node}: {nll:.3f}' for node, nll in zip(row.node_ids, row.mean_correct_candidate_nll)))}</td>"
        f"<td>{row.validation_query_count}</td></tr>"
        for row in artifact.parent_search
    )
    return (
        "<h3>Expected knowledge graph versus learned VAMP graph</h3>"
        "<table><thead><tr><th>symbolic topology</th><th>learned topology</th>"
        f"</tr></thead><tbody>{graph_rows}</tbody></table>"
        "<p>The wrong revision parent is a support problem, but the saved true-parent "
        "counterfactual also failed this revision slice, so it is not a complete causal explanation.</p>"
        "<details open><summary>Validation-only parent search</summary><table><thead><tr>"
        f"<th>task</th><th>selected</th><th>mean correct-candidate NLL by node</th><th>queries</th>"
        f"</tr></thead><tbody>{parent_rows}</tbody></table></details>"
        f"<p>Committed-node drift: <strong>{not artifact.trial.evidence.committed_node_stability.bit_identical and 'detected' or 'zero'}</strong>. "
        f"Allocator peak: {artifact.allocator_peak_bytes / 1024**3:.3f} GiB / "
        f"{artifact.allocator_peak_target_bytes / 1024**3:.0f} GiB.</p>"
    )


def _candidate_figure(diagnostic: CandidateDiagnostic):
    from matplotlib import pyplot

    figure, axis = pyplot.subplots(figsize=(8, 3.4), constrained_layout=True)
    values = [candidate.nll for candidate in diagnostic.candidates]
    colors = [
        "#16a34a" if candidate.correct else "#dc2626" if candidate.predicted else "#94a3b8"
        for candidate in diagnostic.candidates
    ]
    labels = [
        f"{candidate.index}: {candidate.name}\n{candidate.role}"
        for candidate in diagnostic.candidates
    ]
    axis.bar(range(4), values, color=colors)
    axis.set_xticks(range(4), labels)
    axis.set_ylabel("active-token NLL (lower is better)")
    axis.set_title(diagnostic.score.query_id)
    axis.grid(axis="y", alpha=0.25)
    return figure


def _calibration_figure(lab: TinyWorldsLab):
    from matplotlib import pyplot

    facts = [artifact.trial.request.config.facts_per_task for artifact in lab.trials]
    order = sorted(range(len(facts)), key=facts.__getitem__)
    metrics: tuple[tuple[str, Callable[[CalibrationTrialArtifact], float]], ...] = (
        ("frozen direct", lambda artifact: artifact.trial.evidence.frozen_novel_binding.rate),
        ("independent direct", lambda artifact: artifact.trial.evidence.independent_direct_recall.rate),
        ("independent one-hop", lambda artifact: artifact.trial.evidence.independent_one_hop.rate),
        ("revision node", lambda artifact: artifact.trial.evidence.revision_contextual_answer.rate),
        ("paired revision", lambda artifact: artifact.trial.evidence.paired_revision_consistency.rate),
    )
    figure, axis = pyplot.subplots(figsize=(8, 4), constrained_layout=True)
    for label, getter in metrics:
        axis.plot(
            [facts[index] for index in order],
            [getter(lab.trials[index]) for index in order],
            marker="o",
            label=label,
        )
    axis.axhline(0.25, color="#64748b", linestyle="--", linewidth=1, label="4-way chance")
    axis.set_xticks((12, 24, 36))
    axis.set_ylim(-0.04, 1.04)
    axis.set_xlabel("facts exposed per task")
    axis.set_ylabel("candidate accuracy")
    axis.grid(alpha=0.25)
    axis.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    return figure


def _transfer_figure(artifact: CalibrationTrialArtifact, task_id: str):
    from matplotlib import pyplot

    rows = tuple(row for row in artifact.checkpoints if row.task_id == task_id)
    identities = tuple(
        dict.fromkeys((row.stream, row.parent_node_id, row.roles) for row in rows)
    )
    figure, axes = pyplot.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for stream, parent, roles in identities:
        series = tuple(
            row
            for row in rows
            if (row.stream, row.parent_node_id, row.roles) == (stream, parent, roles)
        )
        label = f"{stream} · {parent} · {'/'.join(roles)}"
        axes[0].plot(
            [row.update for row in series],
            [row.validation_candidate_accuracy for row in series],
            marker=".",
            label=label,
        )
        axes[1].plot(
            [row.update for row in series],
            [row.validation_correct_nll for row in series],
            marker=".",
            label=label,
        )
    for axis in axes:
        axis.set_xscale("symlog", linthresh=1)
        axis.set_xlabel("optimizer update")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("validation candidate accuracy")
    axes[1].set_ylabel("validation correct-answer NLL")
    axes[0].set_title(f"{task_id}: accuracy")
    axes[1].set_title(f"{task_id}: NLL")
    axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")
    return figure


def _close_figure(figure: object) -> None:
    from matplotlib import pyplot

    pyplot.close(figure)


__all__ = ["build_tinyworlds_playground"]
