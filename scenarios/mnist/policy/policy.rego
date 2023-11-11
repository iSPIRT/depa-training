package policy

import future.keywords.every
import future.keywords.if

default allowed := false

allowed if {
	valid_key_configuration
}

valid_key_configuration if {
	all_keys_in_contract_referenced
}

all_keys_in_contract_referenced if {
	every dataset in data.datasets {
    	some filesystem in input.azure_filesystems
        filesystem.key.kid == dataset.key.properties.kid
    }
}