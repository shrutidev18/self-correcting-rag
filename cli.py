# This is the front door of your project. The way a user actually talks to the RAG system during development.
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core.pipeline import BasicRAGPipeline


def print_result(result: dict, verbose: bool = True):
    print("\n" + "─" * 60)
    print(f"❓ Question : {result['query']}")
    print(f"💬 Answer   : {result['answer']}")
    print(f"⏱  Latency  : {result['latency_ms']:.0f}ms")
    print(f"🔁 Attempts : {result['attempts']}")

    if verbose and result.get("chunks"):
        print("\n📄 Retrieved chunks:")
        for i, chunk in enumerate(result["chunks"], 1):
            print(f"\n  [{i}] Score: {chunk['score']:.3f} | ID: {chunk['id']}")
            print(f"      {chunk['text'][:200]}...")
    print("─" * 60)


def main():
    parser = argparse.ArgumentParser(description="Self-Correcting RAG — CLI")
    parser.add_argument("question",      nargs="?", help="Question to ask")
    parser.add_argument("--k",           type=int, default=5)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--quiet",       action="store_true")
    args = parser.parse_args()

    print("🚀 Loading pipeline...")
    pipeline = BasicRAGPipeline()
    print("✅ Pipeline ready!\n")

    if args.interactive:
        print("💬 Interactive mode — type 'exit' to quit\n")
        while True:
            try:
                query = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if query.lower() in ("exit", "quit", "q"):
                break
            if not query:
                continue
            result = pipeline.run(query, k=args.k)
            print_result(result, verbose=not args.quiet)

    elif args.question:
        result = pipeline.run(args.question, k=args.k)
        print_result(result, verbose=not args.quiet)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()