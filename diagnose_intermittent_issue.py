import os
import tempfile
import logging
import mlflow
import time
from datetime import datetime

logger = logging.getLogger(__name__)

def diagnose_intermittent_issue(run_id, num_attempts=5):
    """
    Diagnose intermittent MLflow loading issues by running multiple attempts
    and comparing results
    """
    client = mlflow.MlflowClient()
    results = []
    
    print(f"=== Diagnosing intermittent issues for run {run_id} ===")
    print(f"Will attempt {num_attempts} times with 2-second intervals")
    
    for attempt in range(num_attempts):
        print(f"\n--- Attempt {attempt + 1} ---")
        timestamp = datetime.now().isoformat()
        
        result = {
            'attempt': attempt + 1,
            'timestamp': timestamp,
            'success': False,
            'artifacts_found': [],
            'file_sizes': {},
            'error': None
        }
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Try to download artifacts
                try:
                    model_path = client.download_artifacts(run_id, "model-artifact", dst_path=temp_dir)
                    print(f"✓ Downloaded to: {model_path}")
                    
                    # List all files and their sizes
                    for root, dirs, files in os.walk(model_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            rel_path = os.path.relpath(file_path, model_path)
                            file_size = os.path.getsize(file_path)
                            
                            result['artifacts_found'].append(rel_path)
                            result['file_sizes'][rel_path] = file_size
                            print(f"  {rel_path}: {file_size} bytes")
                    
                    # Check for specific model file
                    keras_path = os.path.join(model_path, "data", "model.keras")
                    if os.path.exists(keras_path):
                        size = os.path.getsize(keras_path)
                        print(f"✓ model.keras found: {size} bytes")
                        result['success'] = True
                    else:
                        print("✗ model.keras NOT found at expected path")
                        
                        # Look for any .keras files
                        keras_files = [f for f in result['artifacts_found'] if f.endswith('.keras')]
                        if keras_files:
                            print(f"  But found other .keras files: {keras_files}")
                        else:
                            print("  No .keras files found at all")
                    
                except Exception as download_exc:
                    print(f"✗ Download failed: {download_exc}")
                    result['error'] = str(download_exc)
                    
        except Exception as exc:
            print(f"✗ Overall attempt failed: {exc}")
            result['error'] = str(exc)
        
        results.append(result)
        
        if attempt < num_attempts - 1:
            time.sleep(2)  # Wait between attempts
    
    # Analyze results
    print(f"\n=== Analysis ===")
    successful_attempts = [r for r in results if r['success']]
    failed_attempts = [r for r in results if not r['success']]
    
    print(f"Successful attempts: {len(successful_attempts)}/{num_attempts}")
    print(f"Failed attempts: {len(failed_attempts)}/{num_attempts}")
    
    if successful_attempts and failed_attempts:
        print("\n🚨 INTERMITTENT ISSUE CONFIRMED!")
        print("This suggests:")
        print("  1. Race condition in MLflow artifact storage")
        print("  2. Network/filesystem inconsistency")
        print("  3. Concurrent access issues")
        print("  4. Caching problems")
        
        # Compare successful vs failed attempts
        if successful_attempts:
            success_files = set(successful_attempts[0]['artifacts_found'])
            success_sizes = successful_attempts[0]['file_sizes']
            
            print(f"\nSuccessful attempts typically find: {len(success_files)} files")
            for file_path, size in success_sizes.items():
                print(f"  {file_path}: {size} bytes")
        
        if failed_attempts:
            if failed_attempts[0]['artifacts_found']:
                fail_files = set(failed_attempts[0]['artifacts_found'])
                fail_sizes = failed_attempts[0]['file_sizes']
                
                print(f"\nFailed attempts typically find: {len(fail_files)} files")
                for file_path, size in fail_sizes.items():
                    print(f"  {file_path}: {size} bytes")
                
                # Check for differences
                if successful_attempts:
                    missing_in_failed = success_files - fail_files
                    extra_in_failed = fail_files - success_files
                    
                    if missing_in_failed:
                        print(f"\nFiles missing in failed attempts: {missing_in_failed}")
                    if extra_in_failed:
                        print(f"Extra files in failed attempts: {extra_in_failed}")
                    
                    # Check size differences
                    common_files = success_files & fail_files
                    for file_path in common_files:
                        success_size = success_sizes.get(file_path, 0)
                        fail_size = fail_sizes.get(file_path, 0)
                        if success_size != fail_size:
                            print(f"Size difference for {file_path}: success={success_size}, failed={fail_size}")
            else:
                print("\nFailed attempts find no files at all")
                
    elif len(successful_attempts) == num_attempts:
        print("\n✅ ALL ATTEMPTS SUCCESSFUL - Issue might be resolved or timing-dependent")
    else:
        print("\n❌ ALL ATTEMPTS FAILED - Consistent failure, check MLflow setup")
        
        # Show common errors
        errors = [r['error'] for r in failed_attempts if r['error']]
        if errors:
            print(f"\nCommon errors:")
            for error in set(errors):
                count = errors.count(error)
                print(f"  ({count}x) {error}")
    
    return results


def check_mlflow_backend_store():
    """Check MLflow backend configuration for issues"""
    print("\n=== MLflow Backend Configuration ===")
    
    try:
        client = mlflow.MlflowClient()
        
        # Try to get tracking URI
        tracking_uri = mlflow.get_tracking_uri()
        print(f"Tracking URI: {tracking_uri}")
        
        # Check if using file store vs database
        if tracking_uri.startswith('file://'):
            print("Using file-based backend store")
            store_path = tracking_uri.replace('file://', '')
            if os.path.exists(store_path):
                print(f"✓ Backend store exists: {store_path}")
            else:
                print(f"✗ Backend store missing: {store_path}")
        elif tracking_uri.startswith(('postgresql://', 'mysql://', 'sqlite://')):
            print("Using database backend store")
        else:
            print("Using remote MLflow server")
        
        # Check artifact store
        artifact_uri = mlflow.get_artifact_uri()
        print(f"Default artifact URI: {artifact_uri}")
        
    except Exception as exc:
        print(f"Error checking backend: {exc}")


def suggest_solutions():
    """Suggest solutions based on common intermittent issues"""
    print("\n=== Suggested Solutions ===")
    
    solutions = [
        "1. RETRY LOGIC: Use the robust_model_loader.py with multiple strategies and retries",
        "2. CACHING: Clear MLflow cache if using file-based storage",
        "3. CONCURRENT ACCESS: Ensure only one process accesses MLflow at a time",
        "4. NETWORK ISSUES: If using remote MLflow, check network stability",
        "5. FILESYSTEM: Check disk space and permissions on artifact storage",
        "6. MLflow VERSION: Update to latest MLflow version",
        "7. ALTERNATIVE APPROACH: Store model files directly in artifact store instead of using MLflow model format",
    ]
    
    for solution in solutions:
        print(solution)
    
    print("\nImmediate action:")
    print("Replace your current load_model() function with load_model_robust() from robust_model_loader.py")


# Example usage - replace with your actual run_id
if __name__ == "__main__":
    # You'll need to set your actual run_id here
    # run_id = "your-actual-run-id"
    # results = diagnose_intermittent_issue(run_id)
    
    check_mlflow_backend_store()
    suggest_solutions()
    
    print("\nTo run diagnosis:")
    print("diagnose_intermittent_issue('your-run-id-here', num_attempts=10)")