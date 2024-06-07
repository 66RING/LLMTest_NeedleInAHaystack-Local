run:
	# needlehaystack.run_test --provider local --model_name "/home/ring/Documents/workspace/modules/tinyllama-110M" --document_depth_percents "[50]" --context_lengths "[400]"
	python test.py --provider local --model_name $(MODELS_DIR)/tinyllama-110M  --document_depth_percent_intervals 10 --document_depth_percent_interval_type linear --context_lengths_max 65536 


