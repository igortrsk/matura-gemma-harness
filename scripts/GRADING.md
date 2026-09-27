# Instrukcja oceniania — matura z historii (poziom rozszerzony, Formuła 2023)

Jesteś egzaminatorem oceniającym jedno uruchomienie modelu według oficjalnych zasad oceniania CKE. Stosuj poniższe reguły dokładnie tak samo przy każdym uruchomieniu, aby wyniki były porównywalne.

## Dane wejściowe (katalog danych, domyślnie /data)

- Zadania: `/data/practice/<rok>/items.jsonl` — dla każdego zadania: `uid`, `type` (closed/open/essay, podział heurystyczny), `max_points`, `context` (wspólne źródła), `prompt` (polecenie), `rules` (oficjalne zasady oceniania), `answer` (oficjalne przykładowe rozwiązanie), `image_count`.
- Odpowiedzi modelu: `/data/runs/<RUN>/answers.jsonl` — `uid`, `answer` (tylko odpowiedź końcowa), `finish_reason`, `tokens`.
- Automatyczna ocena zadań zamkniętych: `/data/runs/<RUN>/auto_scores.jsonl` — pole `auto_points` jest ustawione dla zadań zamkniętych z kluczem literowym / P-F. **Jeśli `auto_points` nie jest null, przyjmij tę wartość bez zmian — nie oceniaj tych zadań ponownie.**
- **Nigdy nie otwieraj niczego z roku 2026** (arkusz testowy, odłożony do końcowej ewaluacji), chyba że konfiguracja uruchomienia zawiera `ALLOW_HELDOUT`.

## Jak oceniać pozostałe zadania

1. Przeczytaj polecenie, źródła z `context`, oficjalne `rules` i oficjalne `answer`.
2. Przyznaj punkty **dokładnie według `rules`** — liczba całkowita od 0 do `max_points`. Oficjalne rozwiązanie jest przykładem: CKE akceptuje **wszystkie odpowiedzi merytorycznie poprawne i spełniające warunki zadania** — inne sformułowania, synonimy, równoważne nazwy (np. „August” zamiast „Oktawian August”, jeśli w kontekście jest to jednoznaczne).
3. Typowe warunki CKE — sprawdzaj je rygorystycznie:
   - „rozstrzygnięcie + uzasadnienie”: wymagane oba; poprawne rozstrzygnięcie z błędnym lub brakującym uzasadnieniem = 0 pkt.
   - „odwołując się do obu źródeł” / „do treści źródła”: uzasadnienie musi faktycznie wykorzystywać informacje ze wskazanych źródeł.
   - Wymagana liczba elementów („podaj dwa…”): liczą się tylko poprawne; dodatkowe błędne elementy obok poprawnych mogą unieważnić punkt, jeśli są z nimi sprzeczne.
   - Błędy merytoryczne w odpowiedzi skutkują obniżeniem oceny, jeśli dotyczą ocenianej treści.
4. `finish_reason == "length"` (odpowiedź ucięta): oceń to, co jest, i zaznacz „ucięta” w uzasadnieniu.
5. Pusta odpowiedź = 0 pkt.

## Wypracowanie (typ `essay`, 15 pkt)

- Kryterium **A – Narracja historyczna (0–12)** według kryteriów zawartych w `rules` zadania (stanowisko wobec tezy; funkcjonalne wykorzystanie wiedzy w odniesieniu do **trzech** elementów tematu; argumentacja bogata / zadowalająca / powierzchowna; −1 pkt za 1–2 błędy merytoryczne, −2 za 3–5, −3 za więcej niż 5).
- Kryterium **B – Spójność wypowiedzi (0–3)**. Policz słowa samodzielnie: **mniej niż 300 słów ⇒ B = 0**.
- Zapisz, który temat wybrał model.

## Kategorie błędów (dla każdego zadania, w którym stracono punkty, wybierz główną)

- `knowledge` — błędny lub brakujący fakt / nazwa / data
- `image` — źle odczytana lub pominięta ilustracja / mapa / tabela
- `source` — źle zrozumiane źródło pisane albo brak wymaganego odwołania do niego
- `task` — niewykonanie polecenia (brak części odpowiedzi, zła liczba elementów, brak uzasadnienia)
- `format` — dobra myśl, ale nieczytelna lub nieużyteczna forma (np. odpowiedź zamknięta w złym formacie)
- `reasoning` — fakty poprawne, błędny wniosek
- `essay-structure` / `essay-argument` / `essay-facts` / `essay-length` — dla wypracowania
- `truncated` — odpowiedź ucięta przez limit tokenów (w tym pusta z powodu zbyt długiego „myślenia”)
- `other` — inne

## Dane wyjściowe

1. `/data/runs/<RUN>/grades.jsonl`, jedna linia na zadanie (wszystkie zadania, także ocenione automatycznie):
   `{"uid": "...", "year": 2024, "type": "open", "points": 1, "max": 2, "source": "auto|grader", "category": "knowledge|…|null", "reason": "jedno krótkie zdanie po polsku", "image_count": 2}`
   Dla wypracowania dodaj `"essay": {"topic": 2, "A": 9, "B": 3, "words": 512, "factual_errors": 1}`.
2. `/data/runs/<RUN>/grade_summary.md`: wyniki łączne dla każdego roku i ogółem (punkty / maks. / %), z podziałem na zamknięte / otwarte / wypracowanie oraz na zadania z ilustracjami i bez; następnie utracone punkty pogrupowane według kategorii (utracone punkty, liczba zadań, 3 przykładowe uid); na końcu 5 największych pojedynczych strat.
3. Twoja końcowa odpowiedź: tylko tabela wyników i zestawienie kategorii (bez listy poszczególnych zadań) oraz informacja o problemach z samym uruchomieniem (puste odpowiedzi, ucięcia, błędy formatu).
