for script in scripts/batch_size_discovery_scripts/*.sh; do
    echo "Submitting $script"
    sbatch "$script"
done
