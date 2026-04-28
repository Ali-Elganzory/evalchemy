for script in scripts/eval_scripts/*.sh; do
    echo "Submitting $script"
    sbatch "$script"
done
