"""
Main entry point for the research paper generator.
"""

from langgraph.checkpoint.sqlite import SqliteSaver

from src.agents.graph import create_graph, get_initial_state
from src.utils.config import Config


def main():
    """Run the paper generation workflow."""
    # Example research question
    research_question = "How can I efficiently query connected components in large temporal graphs with scalable and maintainable indices?"

    print("=" * 80)
    print("Graph Theory Research Paper Generator")
    print("=" * 80)
    print(f"\nResearch Question: {research_question}\n")

    # Create checkpointer
    print(f"Creating checkpointer at: {Config.CHECKPOINT_DB_PATH}")
    with SqliteSaver.from_conn_string(Config.CHECKPOINT_DB_PATH) as checkpointer:
        # Build graph
        graph = create_graph(checkpointer=checkpointer)

        # Initialize state
        initial_state = get_initial_state(research_question)

        # Run graph
        config = {"configurable": {"thread_id": "test-run-1"}}

        print("Starting workflow...\n")
        print("-" * 80)

        try:
            # Execute graph
            final_state = graph.invoke(initial_state, config)

            print("-" * 80)
            print("\nWorkflow completed successfully!")
            print(f"Final phase: {final_state.get('current_phase')}")
            print(f"Revision count: {final_state.get('revision_count')}")

            # Show retrieved papers
            retrieved = final_state.get("retrieved_papers", [])
            print(f"\nRetrieved papers: {len(retrieved)}")
            for i, paper in enumerate(retrieved[:5], 1):
                score = paper.get("relevance_score")
                score_str = f"{score:.3f}" if score is not None else "N/A"
                print(f"  {i}. [{score_str}] {paper['title'][:75]}")
            if len(retrieved) > 5:
                print(f"  ... and {len(retrieved) - 5} more")

            # Show errors if any
            if final_state.get("errors"):
                print(f"\nErrors encountered: {len(final_state['errors'])}")
                for error in final_state["errors"]:
                    print(f"  - {error}")

        except Exception as e:
            print(f"\n❌ Error during execution: {e}")
            raise

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
