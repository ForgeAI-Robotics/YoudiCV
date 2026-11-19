#!/usr/bin/env bash
set -euo pipefail

interface_name="eno1"
primary_address="192.168.168.130/24"
piper_alias="192.168.168.162/24"
orangepi_address="192.168.168.120"

addresses=$(ip -br -4 address show dev "${interface_name}")
echo "${addresses}"

for required in "${primary_address}" "${piper_alias}"; do
    if [[ ${addresses} != *"${required}"* ]]; then
        echo "缺少地址：${required}" >&2
        exit 1
    fi
done

ping -c 1 -W 1 "${orangepi_address}" >/dev/null
echo "网络自检通过：双地址存在，OrangePi ${orangepi_address} 可达。"
