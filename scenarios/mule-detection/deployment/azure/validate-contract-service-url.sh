#!/bin/bash

validate_contract_service_url() {
  local url="$1"
  local authority hostport host

  if [[ "$url" != https://* ]]; then
    echo "CONTRACT_SERVICE_URL must be an explicit non-loopback HTTPS endpoint" >&2
    return 1
  fi

  authority="${url#https://}"
  authority="${authority%%/*}"
  authority="${authority%%\?*}"
  authority="${authority%%#*}"
  if [[ -z "$authority" || "$authority" == *"@"* ]]; then
    echo "CONTRACT_SERVICE_URL must contain a valid host without user information" >&2
    return 1
  fi

  hostport="$authority"
  if [[ "$hostport" == \[* ]]; then
    if [[ "$hostport" != *"]"* ]]; then
      echo "CONTRACT_SERVICE_URL must contain a valid host" >&2
      return 1
    fi
    host="${hostport%%]*}"
    host="${host#[}"
  else
    host="${hostport%%:*}"
  fi
  host="${host,,}"
  while [[ "$host" == *. ]]; do
    host="${host%.}"
  done

  if [[ -z "$host" || "$host" == "localhost" || "$host" == *.localhost || \
        "$host" == "host.docker.internal" || "$host" == 127.* || \
        "$host" == "0.0.0.0" || "$host" == "::1" || "$host" == "::" || \
        "$host" == "0:0:0:0:0:0:0:1" || "$host" == ::ffff:127.* ]]; then
    echo "CONTRACT_SERVICE_URL must be an explicit non-loopback HTTPS endpoint" >&2
    return 1
  fi
}
