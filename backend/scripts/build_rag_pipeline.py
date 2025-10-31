#!/usr/bin/env python3
"""
RAG Build Pipeline Orchestrator

Main orchestrator script that coordinates all three phases of the RAG build pipeline:
1. Infrastructure Setup and Validation
2. Document Ingestion and Processing
3. Embedding Generation and Storage

Usage:
    # Run all phases (default)
    python scripts/build_rag_pipeline.py --config rag_pipeline.yaml

    # Run specific phases
    python scripts/build_rag_pipeline.py --config rag_pipeline.yaml --phases infrastructure
    python scripts/build_rag_pipeline.py --config rag_pipeline.yaml --phases ingestion
    python scripts/build_rag_pipeline.py --config rag_pipeline.yaml --phases embedding
    python scripts/build_rag_pipeline.py --config rag_pipeline.yaml --phases infrastructure ingestion

    # Other options
    python scripts/build_rag_pipeline.py --config rag_pipeline.yaml --verbose --output-dir ./reports
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add backend/ to sys.path so "app" imports resolve when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.config import PipelineSpec
from app.pipeline.workflow import RAGWorkflow

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)




async def main() -> int:
    """Main function to run the RAG build pipeline."""
    parser = argparse.ArgumentParser(description="Build RAG ingestion pipeline")
    parser.add_argument("--config", required=True, help="Path to rag_pipeline.yaml (or .json)")
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["infrastructure", "ingestion", "embedding"],
        help="Specific phases to run (default: all phases). Can specify multiple phases."
    )
    parser.add_argument("--output-dir", help="Directory to save reports and outputs")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding generation")
    parser.add_argument("--document-limit", type=int, help="Limit number of documents to process")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load pipeline specification
        spec = PipelineSpec.from_file(args.config)

        # Initialize workflow orchestrator
        workflow = RAGWorkflow(verbose=args.verbose)

        # Determine which phases to run
        phases_to_run = set(args.phases) if args.phases else {"infrastructure", "ingestion", "embedding"}

        logger.info(f"🚀 Starting RAG pipeline with phases: {', '.join(sorted(phases_to_run))}")

        # Run complete pipeline if all phases selected
        if phases_to_run == {"infrastructure", "ingestion", "embedding"}:
            summary = await workflow.run_complete_pipeline(
                spec=spec,
                dry_run=args.dry_run,
                batch_size=args.batch_size,
                document_limit=args.document_limit,
                output_dir=args.output_dir,
            )
            summary.print_summary()
            return 0 if summary.success else 1

        # Run individual phases
        success = True

        # Phase 1: Infrastructure validation
        if "infrastructure" in phases_to_run:
            logger.info("=" * 60)
            logger.info("PHASE 1: INFRASTRUCTURE VALIDATION")
            logger.info("=" * 60)

            infra_result = await workflow.run_infrastructure_check(spec)

            print("\n" + "=" * 60)
            print("INFRASTRUCTURE VALIDATION RESULT")
            print("=" * 60)
            print(f"Success: {'✅ YES' if infra_result.success else '❌ NO'}")

            if infra_result.errors:
                print(f"\n❌ {len(infra_result.errors)} validation errors:")
                for error in infra_result.errors:
                    print(f"   - {error}")
                success = False
            else:
                print("\n✅ All infrastructure components validated successfully!")

            if infra_result.warnings:
                print(f"\n⚠️  {len(infra_result.warnings)} warnings:")
                for warning in infra_result.warnings:
                    print(f"   - {warning}")

        # Phase 2: Document ingestion
        if "ingestion" in phases_to_run:
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 2: DOCUMENT INGESTION")
            logger.info("=" * 60)

            ingestion_summary = await workflow.run_ingestion(spec, dry_run=args.dry_run)

            print("\n" + "=" * 60)
            print("DOCUMENT INGESTION SUMMARY")
            print("=" * 60)
            print(f"Documents Discovered: {ingestion_summary.counters.get('discovered', 0)}")
            print(f"Documents Processed: {ingestion_summary.processed}")
            print(f"Documents Skipped: {ingestion_summary.skipped}")
            print(f"Documents Failed: {ingestion_summary.failed}")
            print(f"Chunks Generated: {ingestion_summary.counters.get('chunking', {}).get('total_chunks', 0)}")

            if ingestion_summary.errors:
                print(f"\n⚠️  {len(ingestion_summary.errors)} errors occurred:")
                for error in ingestion_summary.errors[:5]:
                    print(f"   - {error.get('message', str(error))}")
                if len(ingestion_summary.errors) > 5:
                    print(f"   ... and {len(ingestion_summary.errors) - 5} more errors")
                success = False

        # Phase 3: Embedding generation
        if "embedding" in phases_to_run:
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 3: EMBEDDING GENERATION")
            logger.info("=" * 60)

            embedding_result = await workflow.run_embedding(
                spec,
                batch_size=args.batch_size,
                document_limit=args.document_limit,
            )

            print("\n" + "=" * 60)
            print("EMBEDDING GENERATION SUMMARY")
            print("=" * 60)
            print(f"Success: {'✅ YES' if embedding_result.success else '❌ NO'}")
            print(f"Documents Processed: {embedding_result.documents_processed}")
            print(f"Total Chunks: {embedding_result.total_chunks}")
            print(f"Unique Chunks: {embedding_result.unique_chunks}")
            print(f"Duplicates Removed: {embedding_result.duplicates_removed}")
            print(f"Embeddings Generated: {embedding_result.embeddings_generated}")
            print(f"Embeddings Stored: {embedding_result.embeddings_stored}")
            print(f"Processing Time: {embedding_result.processing_time_seconds:.2f}s")

            if embedding_result.warnings:
                print(f"\n⚠️  {len(embedding_result.warnings)} warnings:")
                for warning in embedding_result.warnings:
                    print(f"   - {warning}")

            if embedding_result.errors:
                print(f"\n❌ {len(embedding_result.errors)} errors occurred:")
                for error in embedding_result.errors:
                    print(f"   - {error}")
                success = False

        # Final status
        logger.info("\n" + "=" * 60)
        if success:
            logger.info("✅ All requested phases completed successfully!")
        else:
            logger.error("❌ Some phases completed with errors")
        logger.info("=" * 60)

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("🛑 Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))