package policy

import future.keywords.every
import future.keywords.if

allowed if {
	all_datasets_in_contract_included
	modeller_filesystem_mounted
	valid_pipeline
}

all_datasets_in_contract_included if {
	every dataset in data.datasets {
		some filesystem in input.azure_filesystems

		# key management service match
		dataset.key.properties.endpoint == filesystem.key.akv.endpoint

		# key identifiers match
		dataset.key.properties.kid == filesystem.key.kid

		# file system mounted at a well-known point
		filesystem.mount_point == concat("", ["/mnt/remote/", dataset.owner])

		# file system must be mounted read only
		filesystem.read_write == false
	}
}

modeller_filesystem_mounted if {
	# expect two additional filesystems (or one if model is instantiated from config in CCR)
	providers = {p | p = data.datasets[_].owner}
	count(input.azure_filesystems) <= count(providers) + 2
	count(input.azure_filesystems) > count(providers)
}

valid_pipeline if {
	data_has_privacy_constraints
	last_stage_is_private_training
	min_privacy_budget_allocated
}

valid_pipeline if {
	not data_has_privacy_constraints
	last_stage_is_training
}

data_has_privacy_constraints if {
	count(data.constraints[_].privacy) > 0
}

last_stage_is_private_training if {
	input.pipeline[count(input.pipeline) - 1].config.is_private == true
}

last_stage_is_training if {
	input.pipeline[count(input.pipeline) - 1].config.is_private == false
}

min_privacy_budget_allocated if {
	threshold = min({t | t = to_number(data.constraints[_].privacy[_].epsilon_threshold)})
	last := count(input.pipeline)
	input.pipeline[last - 1].config.privacy_params.epsilon <= threshold
}
