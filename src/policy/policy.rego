package policy

import future.keywords.every
import future.keywords.if

allowed if {
  all_datasets_in_contract_included
  output_filesystem_mounted
  min_privacy_budget_allocated
}

all_datasets_in_contract_included if {
  every dataset in data.datasets {
    some filesystem in input.azure_filesystems

    # key management service match
    dataset.key.properties.endpoint == filesystem.key.akv.endpoint

    # key identifiers match
    dataset.key.properties.kid == filesystem.key.kid

    # file system mounted at a well-known point
    filesystem.mount_point == concat("", ["/mnt/remote/", dataset.name])

    # file system must be mounted readonly
    filesystem.read_write == false
  }
}

output_filesystem_mounted if {
	# expect two additional filesystems
	count(input.azure_filesystems) == count(data.datasets) + 2
}

min_privacy_budget_allocated if {
  threshold = min({ t | t = to_number(data.constraints[_].privacy[_].epsilon_threshold) })
  input.epsilon_threshold <= threshold
}
