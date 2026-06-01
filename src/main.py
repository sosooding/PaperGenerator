"""
Main entry point for the research paper generator.
"""

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from src.agents.graph import create_graph, get_initial_state
from src.utils.config import Config


def _handle_interrupt(interrupt_value: dict) -> str:
    """Print interrupt prompt and collect user input from stdin."""
    print("\n" + "=" * 80)
    print("[INPUT REQUIRED]")
    print(interrupt_value.get("message", ""))
    print("-" * 40)

    # Paper approval interrupt
    if "papers" in interrupt_value and "gaps" not in interrupt_value:
        for p in interrupt_value["papers"]:
            print(f"  {p['index']:>2}. [{p['relevance_score']:.3f}] ({p['year']}) {p['title'][:70]}")
        print()
        return input("Your choice (Enter = keep all): ").strip()

    # Gap selection interrupt
    if "gaps" in interrupt_value:
        for g in interrupt_value["gaps"]:
            print(f"  {g['index']}. [{g['novelty_score']:.2f}] {g['gap_title']}")
            print(f"      {g['description'][:100]}")
        print()
        return input("Your choice (number): ").strip()

    # Generic fallback
    return input("Your choice: ").strip()


def main():
    """Run the paper generation workflow."""
    research_question = (
        "How can I efficiently query connected components in large temporal graphs "
        "with scalable and maintainable indices?"
    )

    print("=" * 80)
    print("Graph Theory Research Paper Generator")
    print("=" * 80)
    print(f"\nResearch Question: {research_question}\n")

    with SqliteSaver.from_conn_string(Config.CHECKPOINT_DB_PATH) as checkpointer:
        graph = create_graph(checkpointer=checkpointer)
        initial_state = get_initial_state(research_question)
        config = {"configurable": {"thread_id": "run-1"}}

        print("Starting workflow...\n" + "-" * 80)

        try:
            result = graph.invoke(initial_state, config)

            # Resume through any number of interrupt checkpoints
            while True:
                interrupts = result.get("__interrupt__")
                if not interrupts:
                    break
                user_input = _handle_interrupt(interrupts[0].value)
                print("-" * 80)
                result = graph.invoke(Command(resume=user_input), config)

            print("-" * 80)
            print("\nWorkflow completed!")
            print(f"Final phase : {result.get('current_phase')}")
            print(f"Revisions   : {result.get('revision_count')}")

            papers = result.get("retrieved_papers", [])
            print(f"Papers kept : {len(papers)}")
            for i, p in enumerate(papers[:5], 1):
                score = p.get("relevance_score")
                print(f"  {i}. [{score:.3f}] {p['title'][:75]}")
            if len(papers) > 5:
                print(f"  ... and {len(papers) - 5} more")

            decisions = result.get("human_decisions", [])
            if decisions:
                print(f"\nCheckpoints : {len(decisions)}")
                for d in decisions:
                    print(f"  - {d['checkpoint_name']} @ {d['timestamp']}")

            if result.get("errors"):
                print(f"\nErrors ({len(result['errors'])}):")
                for e in result["errors"]:
                    print(f"  - {e}")

        except Exception as exc:
            print(f"\nError during execution: {exc}")
            raise

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
