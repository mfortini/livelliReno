# Fetch-storico: cache parquet flatten al posto di json.gz nested

Sostituire la conversione mensile `.json.gz` → `.parquet` in `tools/fetch-storico.py`: oggi copia il JSON nested (schema instabile per mese), domani produce un parquet flatten tipizzato; poi rigenerare i mesi di cache e verificare parità con `tools/verify-storico-cache.py` **prima** di eliminare i `.json.gz`.

## 1. Problema

- `storico/raw/` contiene i file mensili `YYYY-MM.json.gz` (i parquet nested sono stati cancellati).
- `json_gz_to_parquet()` usa `read_json()` di DuckDB, che inferisce la colonna `data[].vars` come **`STRUCT` per 92 mesi e `MAP` per 51 mesi** (2021-10, 2021-12, 2022-03, 2022-12 e tutto il periodo 2023-01 → 2026-11).
- `_extract_month_parquet()` funziona solo con lo STRUCT → **i 51 mesi MAP restituiscono 0 misure** (verificato: 2023-11 e 2026-11 → levels=0 contro 56903/28316 del percorso `.json.gz`).
- Nota: i per-station `storico/*.parquet` esistenti sono già corretti (prodotti dal percorso `.json.gz`); questa modifica riguarda solo la **cache raw**.

## 2. Formato di destinazione

Una riga per ogni voce `vars.<codice>` di ogni blocco `data[]` di ogni riga JSON (anche se `v` è stringa o null), con metadati denormalizzati:

| colonna | tipo | origine |
|---|---|---|
| `stamp` | VARCHAR | `date` (ISO grezza, es. `2023-11-30T23:45:00Z`) |
| `station_b01019` | VARCHAR | **`data[0].vars.B01019.v` propagato** su tutte le righe del record (non dal blocco corrente) |
| `version` `network` `ident` | VARCHAR | top-level JSON |
| `lon` `lat` | INTEGER | top-level (centesimi di grado) |
| `block_idx` | SMALLINT | indice in `data[]` (0 = metadati) |
| `timerange_0/1/2` | INTEGER | elementi di `timerange` (NULL nel blocco metadati). **INTEGER: il 3° arriva a 86400** |
| `level_0/1/2/3` | INTEGER | elementi di `level` |
| `code` | VARCHAR | chiave `vars` (B13215, B13011, B01019, …) |
| `code_desc` | VARCHAR | decodifica Bufr (`CODE_DESC`; NULL se ignoto) |
| `value_num` | DOUBLE | `v` se numerico; **NULL se stringa o null** |
| `attrs` | JSON | oggetto `a` completo; NULL se assente |

Costo accettato del modello: ~80% delle righe sono del blocco metadati (`block_idx=0`: B04001–06, nome, rete, …). Coerente con “una riga per valore”.

### Decisioni (motivate)

- **Niente `value_str`**: stringhe solo su `B01019` (= `station_b01019`) e `B01194` (= `network`) → `value_num` NULL, nessuna perdita.
- **`attrs` unica colonna JSON**: preserva chiavi `a` future (oggi solo `B33007`).
- **Cache mensile 1:1** (niente multi-mese): 3 mesi in un file → ~18 MB/mese vs ~7.6 MB/mese singolo. Le stazioni si comprimono bene (~1% size); a peggiorare è **`stamp`**: dictionary Parquet **per row group**, più mesi ⇒ più timestamp distinti ⇒ dictionary più grassi duplicati su più RG.
- **`ORDER BY` a cardinalità crescente**: `version, network, ident, level_0..3, timerange_0..2, block_idx, code, station_b01019, lat, lon, stamp` → −43% (7.65 vs 13.2 MB/mese a RG default).
- **`ROW_GROUP_SIZE 1_000_000`**: meno RG ⇒ meno copie del dictionary `stamp` → **~4.3 MB/mese** (ulteriore −44% sul già ordinato). TIMESTAMP/epoch non battono VARCHAR con RG grande.
- Lettura con `columns={...,'data':'JSON'}` → schema **identico** STRUCT/MAP.

### Trial 2023-11

- 20.8M righe; ~20 s/mese con RG 1e6; parità extract livelli+pioggia vs `.json.gz`.
- `timerange` len 3 / `level` len 4 quando presenti; `v` solo num/str/null.
- `a` = `{"B33007":0}` sulle misure mancanti.

## 3. Modifiche codice

### `tools/fetch-storico.py`

1. **`CODE_DESC`**: dict statico Bufr (ignoti → NULL).
2. **`json_gz_to_parquet`**: flatten tipizzato + `ORDER BY` + `COMPRESSION ZSTD` + `ROW_GROUP_SIZE 1000000`. `station_b01019` da `data[0]`. Non cancella il `.json.gz`.
3. **`_extract_month_parquet`**: query piatta su `stamp/station/code/timerange_*/value_num`; rimosso `_parquet_var_codes`.
4. **`download_month`**: dopo download/convert può ancora rimuovere il gz (uso a regime); in migrazione si usa lo script di verify che tiene i gz finché la parità non passa.
5. Help/`tools/README.md`: cache mensile parquet flatten.

### `tools/verify-storico-cache.py` (gate per cancellare i json)

Script dedicato — **unico modo previsto per cancellare i `.json.gz`**:

```bash
# 1) converte tutti i gz → parquet flatten (gz restano)
uv run python tools/verify-storico-cache.py --convert

# 2) parità extract su ogni mese che ha entrambi i file
uv run python tools/verify-storico-cache.py

# 3) solo se exit 0: cancella i gz dei mesi verificati ok
uv run python tools/verify-storico-cache.py --delete-json
```

Cosa controlla, per ogni `YYYY-MM` con `.json.gz` + `.parquet`:

1. **Parità extract** (stazioni livello + pluviometri da `config.json`): `extract_month(json.gz)` vs `extract_month(parquet)` — stessi stamp e valori per ogni stazione (livelli e pioggia).
2. **Sanity schema** (campione): `timerange` len 3 / `level` len 4 sul gz; sul parquet presenza colonne attese e `station_b01019` senza NULL sulle righe con `code` in {B13215,B13011}.
3. Exit code ≠ 0 se un mese fallisce; `--delete-json` **rifiuta** di cancellare se la verifica nello stesso run non è tutta ok (o se si passa solo `--delete-json` senza aver verificato: lo script riverifica sempre prima di cancellare).

Opzioni utili: `--cache`, `--config`, `--from/--to`, `--limit N`, `--convert`, `--force-convert`, `--delete-json`.

## 4. Procedura operativa

0. (Opzionale) pre-check chiavi `a` diverse da B33007 sui gz.
1. `uv run python tools/verify-storico-cache.py --convert` → ~20–25 s/mese.
2. `uv run python tools/verify-storico-cache.py` → parità su tutti i mesi; i 51 ex-MAP devono passare da 0 a valori pieni.
3. `node --check` / `uv run python -c "import tools…"` o `python -m py_compile tools/fetch-storico.py tools/verify-storico-cache.py`.
4. **Solo dopo exit 0**: `uv run python tools/verify-storico-cache.py --delete-json` (~2.3 GB liberati).

## 5. Rischi e limiti

- `timerange`/`level` di lunghezza diversa in mesi non testati → elementi oltre 3/4 persi (coperto dallo schema check; i TR usati dal dashboard sono len-3).
- Codici Bufr ignoti → `code_desc` NULL.
- Stima cache: ~4.3 MB × N mesi (≪ 2.3 GB di gz).
- `storico/*.parquet` per-stazione non vanno rigenerati (sono output, non cache).
