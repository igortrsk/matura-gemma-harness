#!/usr/bin/env bash
# Download official CKE history matura (poziom rozszerzony) papers, source packets and marking guides.
# Source: https://cke.gov.pl — CKE materials are not redistributed in this repo; run this script to fetch them.
set -euo pipefail
DEFAULT_DATA=data; [ -d /data ] && DEFAULT_DATA=/data
OUT="${1:-${DATA_DIR:-$DEFAULT_DATA}/raw/cke}"
B23=https://cke.gov.pl/images/_EGZAMIN_MATURALNY_OD_2023
B15=https://cke.gov.pl/images/_EGZAMIN_MATURALNY_OD_2015/Arkusze_egzaminacyjne
mkdir -p "$OUT/f2023" "$OUT/f2015" "$OUT/informator"

# Formuła 2023 — the current exam format (target)
while read -r name url; do curl -fsSL --retry 3 -o "$OUT/f2023/$name" "$url"; echo "ok f2023/$name"; done <<LIST
2023-arkusz.pdf $B23/Arkusze_egzaminacyjne/2023/Historia/MHIP-R0-100-2305.pdf
2023-zasady.pdf $B23/Arkusze_egzaminacyjne/2023/Historia/MHIP-R0-100-2305-zasady.pdf
2024-arkusz.pdf $B23/Arkusze_egzaminacyjne/2024/Historia/MHIP-R0-100-A-2405-arkusz.pdf
2024-karta.pdf  $B23/Arkusze_egzaminacyjne/2024/Historia/MHIP-R0-100-A-2405-karta.pdf
2024-zasady.pdf $B23/Arkusze_egzaminacyjne/2024/Historia/MHIP-R0-100-2405-zasady.pdf
2025-arkusz.pdf $B23/Arkusze_egzaminacyjne/2025/Historia/MHIP-R0-100-A-2505-arkusz.pdf
2025-karta.pdf  $B23/Arkusze_egzaminacyjne/2025/Historia/MHIP-R0-100-A-2505-karta.pdf
2025-zasady.pdf $B23/Arkusze_egzaminacyjne/2025/zasady_oceniania/MHIP-R0-100-2505-zasady.pdf
2026-arkusz.pdf $B23/Arkusze_egzaminacyjne/2026/Historia/MHIP-R0-100-A-2605-arkusz.pdf
2026-karta.pdf  $B23/Arkusze_egzaminacyjne/2026/Historia/MHIP-R0-100-A-2605-karta.pdf
2026-zasady.pdf $B23/Arkusze_egzaminacyjne/2026/Historia/MHIP-R0-100-2605-zasady.pdf
LIST

# Informator 2025/2026 — official sample tasks with solutions
curl -fsSL --retry 3 -o "$OUT/informator/informator-2025-2026.pdf" "$B23/Informatory/2024/Informator_EM2025_historia_2025_2026.pdf" && echo "ok informator"

# Formuła 2015 — older format, extra practice/training material
while read -r name url; do curl -fsSL --retry 3 -o "$OUT/f2015/$name" "$url"; echo "ok f2015/$name"; done <<LIST
2015-arkusz.pdf $B15/2015/formula_od_2015/MHI-R1_1P-152.pdf
2015-zasady.pdf $B15/2015/formula_od_2015/odpowiedzi/MHI-R1-N.pdf
2016-arkusz.pdf $B15/2016/formula_od_2015/MHI-R1_1P-162.pdf
2016-zasady.pdf $B15/2016/formula_od_2015/zasady_oceniania/MHI-R1-N.pdf
2017-arkusz.pdf $B15/2017/formula_od_2015/historia/MHI-R1_1P-172.pdf
2017-zasady.pdf $B15/2017/formula_od_2015/zasady_oceniania/MHI-R1-N.pdf
2018-arkusz.pdf $B15/2018/formula_od_2015/historia/MHI-R1_1P-182.pdf
2018-zasady.pdf $B15/2018/formula_od_2015/Zasady_oceniania/MHI-R1_1P-182_zasady_oceniania.pdf
2019-arkusz.pdf $B15/2019/formula_od_2015/historia/MHI-R1_1P-192.pdf
2019-zasady.pdf $B15/2019/formula_od_2015/Zasady_oceniania/MHI-R1_1P-192_model.pdf
2020-arkusz.pdf $B15/2020/formula_od_2015/historia/MHI-R1_1P-202.pdf
2021-arkusz.pdf $B15/2021/Historia/poziom_rozszerzony/EHIP-R0-100-2105.pdf
2022-arkusz.pdf $B15/2022/Historia/poziom_rozszerzony/EHIP-R0-100-2205.pdf
2023-arkusz.pdf $B15/2023/Historia/EHIP-R0-100-2305.pdf
2023-zasady.pdf $B15/2023/Historia/EHIP-R0-100-2305-zasady.pdf
LIST
