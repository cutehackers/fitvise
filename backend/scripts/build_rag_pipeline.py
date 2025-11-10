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
    
    # Run  specific phase with --dry-run
    python scripts/build_rag_pipeline.py --config rag_pipeline.yaml --phases ingestion --dry-run 

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
from app.infrastructure.database.database import async_session_maker

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
        logger.info("🔧 Verbose logging enabled - DEBUG level activated")
        print(f"📝 Running with verbose logging: args.verbose={args.verbose}")
    else:
        logger.info("ℹ️ Running with normal logging (use --verbose for detailed logs)")

    try:
        # Load pipeline specification
        spec = PipelineSpec.from_file(args.config)

        # Create database session for the pipeline
        async with async_session_maker() as session:
            # Initialize workflow orchestrator with database session
            workflow = RAGWorkflow(verbose=args.verbose, session=session)

            # Determine which phases to run
            phases = set(args.phases) if args.phases else {"infrastructure", "ingestion", "embedding"}

            logger.info(f"🚀 Starting RAG pipeline with phases: {', '.join(sorted(phases))}")

            # Run complete pipeline if all phases selected
            if phases == {"infrastructure", "ingestion", "embedding"}:
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
            if "infrastructure" in phases:
                logger.info("=" * 60)
                logger.info("PHASE 1: INFRASTRUCTURE VALIDATION")
                logger.info("=" * 60)

                infra_report = await workflow.run_infrastructure_check(spec)

                print("\n" + "=" * 60)
                print("INFRASTRUCTURE VALIDATION RESULT")
                print("=" * 60)
                print(f"Success: {'✅ YES' if infra_report.success else '❌ NO'}")

                if infra_report.errors:
                    print(f"\n❌ {len(infra_report.errors)} validation errors:")
                    for error in infra_report.errors:
                        print(f"   - {error}")
                    success = False
                else:
                    print("\n✅ All infrastructure components validated successfully!")

                if infra_report.warnings:
                    print(f"\n⚠️  {len(infra_report.warnings)} warnings:")
                    for warning in infra_report.warnings:
                        print(f"   - {warning}")

            # Phase 2: Document ingestion
            if "ingestion" in phases:
                logger.info("\n" + "=" * 60)
                logger.info("PHASE 2: DOCUMENT INGESTION")
                logger.info("=" * 60)

                ingestion_report = await workflow.run_ingestion(spec, dry_run=args.dry_run)

                print("\n" + "=" * 60)
                print("DOCUMENT INGESTION SUMMARY")
                print("=" * 60)
                print(f"Documents discovered: {ingestion_report.discovered}")
                print(f"Documents processed: {ingestion_report.processed}")
                print(f"Documents skipped: {ingestion_report.skipped}")
                print(f"Documents failed: {ingestion_report.failed}")
                print(f"Chunks generated: {ingestion_report.chunking_summary.get('total_chunks', 0)}")

                if ingestion_report.errors:
                    print(f"\n⚠️  DOCUMENT INGESTION ERRORS ({len(ingestion_report.errors)} issues):")
                    for i, error in enumerate(ingestion_report.errors[:5], 1):
                        error_msg = error.get('message', str(error))
                        error_type = error.get('type', 'Unknown')
                        print(f"   {i}. [{error_type}] {error_msg}")

                        # Provide actionable suggestions for common errors
                        if "path does not exist" in str(error_msg).lower():
                            print(f"      💡 SOLUTION: Check the 'documents.path' in your config file")
                        elif "permission" in str(error_msg).lower():
                            print(f"      💡 SOLUTION: Check file permissions for the document path")
                        elif "corrupted" in str(error_msg).lower() or "invalid" in str(error_msg).lower():
                            print(f"      💡 SOLUTION: Verify the document file is not corrupted")

                    if len(ingestion_report.errors) > 5:
                        print(f"   ... and {len(ingestion_report.errors) - 5} more errors")

                    print(f"\n📊 SUMMARY: {ingestion_report.processed} processed, {ingestion_report.failed} failed, {ingestion_report.skipped} skipped")
                    success = False

            # Phase 3: Embedding generation
            if "embedding" in phases:
                logger.info("\n" + "=" * 60)
                logger.info("PHASE 3: EMBEDDING GENERATION")
                logger.info("=" * 60)

                embedding_report = await workflow.run_embedding(
                    spec,
                    batch_size=args.batch_size,
                    document_limit=args.document_limit,
                )

                print("\n" + "=" * 60)
                print("EMBEDDING GENERATION SUMMARY")
                print("=" * 60)
                print(f"Success: {'✅ YES' if embedding_report.success else '❌ NO'}")
                print(f"Documents Processed: {embedding_report.phase_result.documents_processed}")
                print(f"Total Chunks: {embedding_report.phase_result.total_chunks}")
                print(f"Unique Chunks: {embedding_report.phase_result.unique_chunks}")
                print(f"Duplicates Removed: {embedding_report.phase_result.duplicates_removed}")
                print(f"Embeddings Generated: {embedding_report.phase_result.embeddings_generated}")
                print(f"Embeddings Stored: {embedding_report.phase_result.embeddings_stored}")
                print(f"Processing Time: {embedding_report.phase_result.processing_time_seconds:.2f}s")

                if embedding_report.phase_result.warnings:
                    print(f"\n⚠️  {len(embedding_report.phase_result.warnings)} warnings:")
                    for warning in embedding_report.phase_result.warnings:
                        print(f"   - {warning}")

                if embedding_report.phase_result.errors:
                    print(f"\n❌ EMBEDDING GENERATION ERRORS ({len(embedding_report.phase_result.errors)} issues):")
                    for i, error in enumerate(embedding_report.phase_result.errors[:5], 1):
                        print(f"   {i}. {error}")

                        # Provide actionable suggestions for common embedding errors
                        error_str = str(error).lower()
                        if "trust_remote_code" in error_str:
                            print(f"      💡 SOLUTION: This should be fixed now with trust_remote_code=True")
                        elif "memory" in error_str or "cuda" in error_str:
                            print(f"      💡 SOLUTION: Reduce batch_size or free up system memory")
                        elif "connection" in error_str or "timeout" in error_str:
                            print(f"      💡 SOLUTION: Check Weaviate connection and network connectivity")
                        elif "model" in error_str:
                            print(f"      💡 SOLUTION: Ensure embedding model is downloaded and accessible")

                    if len(embedding_report.phase_result.errors) > 5:
                        print(f"   ... and {len(embedding_report.phase_result.errors) - 5} more errors")

                    success = False

            # Final status
            logger.info("\n" + "=" * 60)
            if success:
                logger.info("✅ All requested phases completed successfully!")
            else:
                logger.error("❌ Pipeline completed with ERRORS - See details above for solutions")
                logger.error("💡 Most common fixes:")
                logger.error("   • Check 'documents.path' in your config file exists")
                logger.error("   • Ensure Weaviate is running: docker-compose up -d weaviate")
                logger.error("   • Verify document files are readable and not corrupted")
                logger.error("   • Try smaller batch sizes if memory errors occur")
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