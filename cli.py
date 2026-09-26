import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core.pipeline import BasicRAGPipeline


def print_result(result: dict, verbose: bool = True):
    print(f"\nQ: {result['query']}")
    print(f"A: {result['answer']}")
    print(f"\nlatency  : {result['latency_ms']:.0f}ms")
    print(f"attempts : {result['attempts']}")

    if result.get("reformulated_query"):
        print(f"rewrite  : {result['reformulated_query']}")

    if verbose and result.get("chunks"):
        print("\nchunks:")
        for i, chunk in enumerate(result["chunks"], 1):
            print(f"  [{i}] score={chunk['score']:.3f} | {chunk['text'][:150]}...")


def main():
    parser = argparse.ArgumentParser(description="RETRACE — Self-Correcting RAG CLI")
    parser.add_argument("question", nargs="?", help="question to ask")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print("loading pipeline...")
    pipeline = BasicRAGPipeline()
    print("ready\n")

    if args.interactive:
        print("interactive mode — type 'exit' to quit\n")
        while True:
            try:
                query = input("you: ").strip()
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