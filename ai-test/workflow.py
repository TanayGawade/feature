
# This script demonstrates the complete end-to-end AI solution for customer churn prediction.
# It runs through all stages: data preparation, model training, evaluation, and API setup.



import os
import sys
import subprocess
import time

def run_script(script_path, description):
    """Run a Python script and handle any errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    
    try:
        # Use the full Python path
        python_path = r"C:\Users\tanay\AppData\Local\Programs\Python\Python313\python.exe"
        result = subprocess.run([python_path, script_path], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    """Main workflow execution"""
    print("🚀 Customer Churn Prediction - Complete Workflow")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('data/churn_data.csv'):
        print("❌ Error: churn_data.csv not found in data/ directory")
        print("Please ensure you're running this from the project root directory")
        return False
    
    # Step 1: Data Preparation
    if not run_script('src/data_prep.py', 'Data Preparation'):
        print("❌ Data preparation failed")
        return False
    
    # Step 2: Model Training
    if not run_script('src/train.py', 'Model Training'):
        print("❌ Model training failed")
        return False
    
    # Step 3: Model Evaluation
    if not run_script('src/evaluate.py', 'Model Evaluation'):
        print("❌ Model evaluation failed")
        return False
    
    # Step 4: Run Unit Tests
    print(f"\n{'='*60}")
    print("Running Unit Tests")
    print(f"{'='*60}")
    
    try:
        python_path = r"C:\Users\tanay\AppData\Local\Programs\Python\Python313\python.exe"
        result = subprocess.run([python_path, '-m', 'pytest', 'tests/', '-v'], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Some tests failed:")
        print(e.stdout)
        print(e.stderr)


    
    print(f"\n🎉 Workflow completed successfully!")
    print("The customer churn prediction system is ready for use.")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ All tasks completed successfully!")
    else:
        print("\n❌ Workflow failed. Please check the errors above.")
        sys.exit(1) 
