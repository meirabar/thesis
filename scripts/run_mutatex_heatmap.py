import subprocess
import os

def run_mutatex_and_heatmap(protein_id, mutation_list, results_dir, templates_dir, foldx_exec, rotabase):
    pdb_path = f"/home/labs/rudich/meiray/meirab/foldx/{protein_id}.pdb"
    mutation_list_path = os.path.join(templates_dir, "mutation_list.txt")

    # Construct the command without the unnecessary path to the 'mutatex' directory
    mutatex_cmd = ['mutatex',
        pdb_path,
        "-m", mutation_list_path,
        "-f", "suite5",
        "-R", os.path.join(templates_dir, "foldxsuite4/repair_runfile_template.txt"),
        "-M", os.path.join(templates_dir, "foldxsuite4/mutate_runfile_template.txt"),
        "-I", os.path.join(templates_dir, "foldxsuite4/interface_runfile_template.txt"),
        "-x", foldx_exec,
        "-b", rotabase
    ]
    
    print("Running mutatex with command:", " ".join(mutatex_cmd))

    # Execute the command and capture the output
    result = subprocess.run(mutatex_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error running mutatex:")
        print(result.stderr)
        return

    print("Mutatex output:")
    print(result.stdout)


# Replace the paths with your actual setup
protein_id = "1W4F"  # Example protein ID
mutation_list = "/home/labs/rudich/meiray/meirab/foldx/mutatex/templates/mutation_list.txt"
results_dir = "/home/labs/rudich/meiray/meirab/foldx/mutatex/results/mutation_ddgs"
templates_dir = "/home/labs/rudich/meiray/meirab/foldx/mutatex/templates"
foldx_exec = "/home/labs/rudich/meiray/meirab/foldx/foldx_20241231"
rotabase = "/home/labs/rudich/meiray/meirab/foldx/rotabase.txt"

# Run the function
run_mutatex_and_heatmap(protein_id, mutation_list, results_dir, templates_dir, foldx_exec, rotabase)
