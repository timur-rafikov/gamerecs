#!/usr/bin/env bash
# Сборка для Render (LFS-файлы Render подтягивает сам при клонировании). annoy по умолчанию собирается с -march=native: на машине сборки это
# одни инструкции, на рабочем инстансе другие, и процесс падает с SIGILL (код 132).
# Поэтому собираем annoy из исходников с базовым x86-64.
set -e
pip install $(grep -v '^annoy' requirements.txt)

pip uninstall -y annoy || true
rm -rf /tmp/annoy_src && mkdir /tmp/annoy_src
pip download annoy==1.17.3 --no-binary :all: --no-deps -d /tmp/annoy_src
tar xzf /tmp/annoy_src/annoy-1.17.3.tar.gz -C /tmp/annoy_src
sed -i.bak 's/-march=native/-march=x86-64/' /tmp/annoy_src/annoy-1.17.3/setup.py
grep -n "march" /tmp/annoy_src/annoy-1.17.3/setup.py
pip install /tmp/annoy_src/annoy-1.17.3
