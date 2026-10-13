from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY from a local .env file, if present

from rag import RAG  # noqa: E402  (imported after load_dotenv on purpose)


def main():
    print("Loading documents and building the index...")
    rag = RAG("data")
    print(f"Ready: {len(rag.chunks)} chunks indexed. Type 'exit' to quit.\n")

    while True:
        question = input("Ask: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        answer, hits = rag.answer(question)
        print(f"\n{answer}\n")
        print("Sources:", ", ".join(sorted({source for source, _, _ in hits})), "\n")


if __name__ == "__main__":
    main()
