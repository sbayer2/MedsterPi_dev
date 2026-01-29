from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from medster.agent import Agent
from medster.utils.intro import print_intro
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory


def main():
    print_intro()
    agent = Agent(verbose=True)

    # Create a prompt session with history
    session = PromptSession(history=InMemoryHistory())

    while True:
        try:
            # Prompt the user for input
            query = session.prompt("medster>> ")
            if query.lower() in ["exit", "quit"]:
                print("Session ended. Goodbye!")
                break
            if query:
                # Run the clinical analysis agent
                result = agent.run(query)
                if result:
                    print(f"\n{result['response']}\n")
                    usage = result.get("usage", {})
                    print(f"[Tokens: {usage.get('input_tokens', 0):,} in / "
                          f"{usage.get('output_tokens', 0):,} out / "
                          f"{usage.get('total_tokens', 0):,} total | "
                          f"{usage.get('iterations', 0)} iterations]\n")
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended. Goodbye!")
            break


if __name__ == "__main__":
    main()
