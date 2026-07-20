import os
import time
import subprocess

def main():
    results_dir = "outputs/results"
    arms = ["arm0", "arm1", "arm2", "arm3"]
    cohorts = ["pretreat_m2b_test", "brainmetshare", "brainmetshare_test", "ucsf_bmsr"]
    
    expected_files = []
    for arm in arms:
        for cohort in cohorts:
            expected_files.append(f"{arm}_{cohort}_summary.json")
            expected_files.append(f"{arm}_{cohort}_patients.json")
            
    print(f"Monitoring folder '{results_dir}' for {len(expected_files)} files...")
    
    while True:
        missing = []
        for filename in expected_files:
            filepath = os.path.join(results_dir, filename)
            if not os.path.exists(filepath):
                missing.append(filename)
                
        if len(missing) == 0:
            print("\nAll expected evaluation outputs found! Regenerating paper tables...")
            time.sleep(2)
            try:
                # Run generate_paper_tables.py as a module
                result = subprocess.run(
                    ["C:\\Users\\MSI\\miniconda3\\envs\\dr_detection\\python.exe", "-m", "scripts.generate_paper_tables"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print("Tables generated successfully!")
                print(result.stdout)
                
                # Write stdout to a file so we can view it
                with open(os.path.join(results_dir, "auto_generated_latex.txt"), "w") as f:
                    f.write(result.stdout)
            except Exception as e:
                print(f"Error running generate_paper_tables.py: {str(e)}")
            break
            
        print(f"Missing {len(missing)}/32 files. Waiting 60 seconds...")
        time.sleep(60)

if __name__ == "__main__":
    main()
