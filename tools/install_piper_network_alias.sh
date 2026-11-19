#!/usr/bin/env bash
set -euo pipefail

connection_name="Wired connection 1"
interface_name="eno1"
primary_address="192.168.168.130/24"
piper_alias="192.168.168.162/24"
orangepi_address="192.168.168.120"

if [[ ${EUID} -ne 0 ]]; then
    echo "请使用 sudo 运行：sudo $0" >&2
    exit 1
fi

if ! command -v nmcli >/dev/null 2>&1; then
    echo "未找到 nmcli；此安装程序要求 NetworkManager。" >&2
    exit 1
fi

if ! nmcli -t -f NAME connection show | grep -Fxq "${connection_name}"; then
    echo "未找到 NetworkManager 连接：${connection_name}" >&2
    exit 1
fi

saved_addresses=$(nmcli -g ipv4.addresses connection show "${connection_name}")
if ! grep -Fq "${primary_address}" <<<"${saved_addresses}"; then
    echo "拒绝修改：${connection_name} 未包含预期主地址 ${primary_address}" >&2
    exit 1
fi

if grep -Fq "${piper_alias}" <<<"${saved_addresses}"; then
    echo "持久辅助地址已存在：${piper_alias}"
else
    nmcli connection modify "${connection_name}" +ipv4.addresses "${piper_alias}"
    echo "已写入持久辅助地址：${piper_alias}"
fi

# Apply the alias immediately without cycling the interface. This is
# idempotent and keeps the existing .130 address and routes intact.
if ! ip -4 address show dev "${interface_name}" | grep -Fq "inet ${piper_alias}"; then
    ip address add "${piper_alias}" dev "${interface_name}"
    echo "已将辅助地址应用到当前网卡。"
fi

echo "保存的 NetworkManager 地址："
nmcli -g ipv4.addresses connection show "${connection_name}"
echo "当前 ${interface_name} 地址："
ip -br -4 address show dev "${interface_name}"

if ping -c 1 -W 1 "${orangepi_address}" >/dev/null; then
    echo "OrangePi ${orangepi_address}：可达"
else
    echo "警告：OrangePi ${orangepi_address} 当前不可达。" >&2
    exit 2
fi

echo "安装完成；NetworkManager 将在以后每次启动时自动配置 ${piper_alias}。"
