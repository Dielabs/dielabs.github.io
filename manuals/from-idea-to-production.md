---
layout: default
title: From Idea to Production
sitemap: false
---

<!-- noindex: temporary unlisted page -->
<meta name="robots" content="noindex, nofollow">

# From Idea to Production

Framework di sizing e validazione per infrastruttura di inferenza LLM.

---

## Struttura

Il framework definisce una sequenza ordinata di 11 passi che porta da un'esigenza di business a un sistema di inferenza dimensionato e validato empiricamente. La sequenza non è strettamente lineare: i passi 5–9 sono interdipendenti e si percorrono iterativamente, ma la narrazione resta lineare per garantire che ogni decisione sia tracciabile a un input verificato.

```
1. Use case
       │
       ▼
   ┌───────────────────────────────────┐
   │  2. Workload profile (forma)      │
   │  3. Traffic profile (volume)      │ ← input cliente
   │  4. SLO (contratto)               │
   └───────────────────────────────────┘
       │
       ▼
   ┌───────────────────────────────────┐
   │  5. Model architecture            │
   │  6. Sizing                        │
   │  7. Runtime architecture          │ ← risposta architetturale
   │  8. Hardware & fabric             │
   │  9. Serving stack                 │
   └───────────────────────────────────┘
       │
       ▼
   ┌───────────────────────────────────┐
   │ 10. Baseline & Benchmarks         │
   │ 11. Scaling decision / iterazione │ ← validazione + iterazione
   └───────────────────────────────────┘
```

I passi 1–4 sono input dal cliente. I passi 5–9 sono la risposta architetturale. I passi 10–11 chiudono il ciclo con misurazione e scaling consapevole.

> **Workload profile vs Traffic profile.** Il workload profile (passo 2) è la **distribuzione delle richieste** — cosa elabori. Il traffic profile (passo 3) è la **serie temporale dell'arrivo** — quanto ne elabori nel tempo. Sono due oggetti matematici distinti, si stimano con tecniche diverse, e insieme allo SLO (passo 4) formano la triade di input cliente al sizing.

---

### 1. Use case

L'esigenza del cliente da soddisfare.

|Campo|Descrizione|
|---|---|
|Opportunità|Il bisogno, il problema, la necessità del cliente|
|Attore|Chi consuma l'output (B2C, B2B, interno, sistema)|
|Tipo di task|Generation, classification, RAG, agentic, summarization, code|
|Deployment context|Greenfield / Brownfield|
|Constraints|Funzionali, non funzionali, operativi, tecnici, economici|
|Criteri di successo|KPI di business (quelli tecnici sono nel passo SLO)|

**Deployment context:**

- **Greenfield** — nessun vincolo infrastrutturale preesistente, libertà su hardware, fabric, stack, data center
- **Brownfield** — l'inferenza si innesta su infrastruttura esistente (GPU già presenti, fabric definito, orchestratore in uso, vincoli di footprint/power)

---

### 2. Workload profile (forma della richiesta)

La forma delle richieste: cosa il sistema deve elaborare, indipendentemente da quante ne arrivano nel tempo. È una **distribuzione**, ereditata dal workload reale, analizzata da log, o ipotizzata in maniera attenta.

|Campo|Misurazione|
|---|---|
|Context length|Massimo e medio|
|Input tokens|Distribuzione (p50, p95)|
|Output tokens|Distribuzione (p50, p95)|
|Rapporto prefill/decode|Derivato dal rapporto input/output|

> Dimensiona sempre sul massimo / p95 (_abbastanza_ massimo). Il workload profile è quasi sempre una stima al primo giro: si raffina nel passo 10 con il benchmark sul traffico reale.

---

### 3. Traffic profile (volume nel tempo)

Il volume delle richieste e la sua distribuzione temporale. È una **serie temporale**: descrive _quante_ richieste arrivano, _quando_, e _come_ si distribuiscono nel tempo. È la sezione densa della discovery: dove si concentra la negoziazione esplicita con il cliente sul carico atteso.

#### 3.1 Pattern di traffico

Come si distribuisce il carico nel tempo:

- **Costante** — RPS stabile, tipico di sistemi automatizzati o B2B con uso continuativo
- **Bursty** — picchi concentrati su finestre brevi, tipico di workload utente-facing con eventi scatenanti
- **Batch** — finestre di elaborazione massiva alternate a periodi di idle, tipico di pipeline notturne
- **Misto** — combinazione (es. carico baseline + burst occasionali)

Il pattern di traffico determina come si dimensiona l'headroom (3.5) e quale autoscaling sarà necessario al passo 11.

#### 3.2 Concorrenza

Il pattern di traffico nel tempo sul serving diventa concorrenza — il numero di richieste attive contemporanee che l'architettura deve gestire. È il **vincolo dimensionante** che entra direttamente nel passo 6 (sizing).

Il cliente può non fornirtela direttamente, consegnandoti invece una grandezza più vicina al suo modello mentale di business: utenti collegati, RPS target, volume giornaliero. La concorrenza si deriva da una di queste.

**Ordine di preferenza delle fonti** (dalla più dichiarata alla più stimata):

|Ordine|Input dal cliente|Affidabilità|
|---|---|---|
|1|Richieste concorrenti al serving|Diretto, nessuna derivazione|
|2|RPS target + durata E2E|Calcolo via Little|
|3|Utenti contemporanei al picco|Calcolo via duty cycle|
|4|Volume giornaliero/orario|Stima del picco + una delle vie sopra|

> **Nota sulla dipendenza apparente tra passo 3 e passo 4 (SLO).** Il caso 2 usa `durata_E2E`, che è uno SLO formalizzato al passo 4. La separazione tra traffic profile e SLO è didattica: nella discovery con il cliente i due passi si conducono in un'unica conversazione iterativa, perché RPS, durata E2E e concorrenza sono tre numeri che devono quadrare insieme. Qui anticipiamo la variabile per chiudere il calcolo, sapendo che sarà formalizzata subito dopo.

##### Caso 1 — Concorrenza dichiarata

Il cliente ha telemetria del workload esistente o ha già fatto capacity planning. Si usa il valore direttamente, eventualmente sovrascrivendo la sua p95 se la distribuzione mostra code lunghe.

##### Caso 2 — Da RPS a concorrenza (legge di Little)

Tipico quando il cliente esprime il requisito come throughput: _"50 req/s al picco"_. La concorrenza si deriva da RPS e durata media end-to-end della risposta — che è uno SLO target del passo 4.

```
concorrenza = RPS × durata_E2E
```

Esempio: 50 req/s con durata E2E media di 4s → 200 richieste concorrenti al serving.

_I tre numeri (RPS, durata, concorrenza) devono quadrare. Vanno tenuti allineati esplicitamente con il cliente prima di considerare il sizing concluso: se la durata E2E target cambia al passo 4, la concorrenza cambia._

##### Caso 3 — Da utenti contemporanei a concorrenza (duty cycle)

Tipico nei progetti utente-facing (chat, assistant, RAG): il cliente fornisce utenti contemporanei al picco, non richieste in volo al serving. La differenza è il **duty cycle** — la frazione di tempo in cui un utente collegato sta effettivamente generando carico (richiesta in volo). Valori bassi indicano workload con molto idle tra una richiesta e l'altra; valori alti indicano workload continuativi.

```
concorrenza = utenti_contemporanei × duty_cycle
```

**Duty cycle indicativi:**

|Workload|Duty cycle|
|---|---|
|Chat conversazionale|30%|
|Q&A breve / RAG|15–20%|
|Agentic / risposte lunghe|50–60%|
|Batch processing|100%|

Esempio: 100 utenti contemporanei in chat conversazionale → 30 richieste concorrenti al serving.

_Il duty cycle è il parametro più fragile della catena: dipende dal comportamento degli utenti, non dalla tecnologia. Va dichiarato esplicitamente come assunzione e validato in benchmark._

##### Caso 4 — Da volume giornaliero a concorrenza

Il caso peggiore: il cliente fornisce solo volumi aggregati ("1M richieste/giorno"). Serve un doppio passaggio: stima del picco (RPS o utenti contemporanei) a partire dal volume, poi una delle derivazioni sopra. Tipicamente si applica un fattore di concentrazione del traffico per ottenere l'RPS di picco, da cui si procede con Little.

**Fattori di concentrazione indicativi:**

|Workload|Concentrazione del volume giornaliero|
|---|---|
|Batch overnight|100% del volume in N ore di finestra notturna|
|B2C utente-facing|30–50% del volume in 2–3 ore di picco|
|B2B / internal tools|Volume distribuito nell'orario lavorativo (8h) con picco mattutino|
|Sistemi automatizzati / API|Spesso piatto, ma con burst su trigger esterni|

_A questo livello di stima, l'incertezza è alta. Vale la pena dichiararla al cliente e mettere headroom generoso, in attesa di telemetria reale._

#### 3.3 Headroom di provisioning

Margine percentuale che applichiamo sopra la **concorrenza** calcolata, per assorbire crescita futura e picchi sopra il p95. Si dimensiona sul **massimo** dei due contributi, non sulla somma.

L'headroom si applica a valle, sulla concorrenza già calcolata, non sugli utenti o sull'RPS a monte. Mantiene il calcolo lineare e tracciabile.

```
concorrenza_target = concorrenza_derivata × (1 + headroom in percentuale)
```

#### 3.4 Glossario di concorrenza

|Termine|Significato|Origine|
|---|---|---|
|`concorrenza_derivata`|Carico stimato a partire dagli input cliente|Output 3.2|
|`concorrenza_target`|Derivata + headroom — input al passo 6|Output 3.3|
|`concorrenza_massima` (Cr)|Capacità max per replica — capacità sostenibile|Output 6a, validato al passo 10|

---

### 4. SLO (contratto di servizio)

Il contratto di servizio: latenza, disponibilità, e — derivato dai primi due — lo scenario di validazione che governa il passo 10.

#### 4.1 SLO di latenza e disponibilità

|Campo|Descrizione|
|---|---|
|TTFT|Time To First Token (p95 o p99)|
|TPOT o ITL|Time Per Output Token o Inter-Token Latency (p95)|
|E2E|Durata end-to-end della risposta (p95)|
|Disponibilità|Uptime target (es. 99.9%), tolleranza ai guasti, ridondanza|
|SLO di throughput _(opzionale)_|Token/s aggregati, tipicamente medio o p50 — non indispensabile in workload utente-facing|

> **I tre SLO di latenza non sono indipendenti.** Il budget E2E vincola TTFT e ITL via:
>
> E2E ≈ TTFT + ITL × output_p95
>
> Negoziare uno dei tre con il cliente significa toccare anche gli altri. Esempio: E2E target 4s, output p95 = 300 token, TTFT target 1s → budget ITL = (4 − 1) / 300 ≈ 10 ms/token. Se la GPU candidata fa 15 ms/token sul decode, il sizing del passo 6 non chiude — la conversazione torna qui per rinegoziare TTFT, E2E, o ridurre output_p95.

#### 4.2 Profili di SLO

Il mix di SLO dichiarati in 4.1, combinato con il traffic profile del passo 3, determina lo **scenario di validazione** che governa il passo 10.

|Scenario|SLO dominanti|
|---|---|
|**Latency-first**|TTFT e/o E2E p95 stringenti, output utente-facing|
|**Throughput-first**|€/Mtok, throughput aggregato, no SLO p95 stringenti (batch, async)|
|**Full envelope**|Doppia metrica di business, cluster condiviso, sizing iniziale senza telemetria|

Lo scenario è un _output_ del passo 4: descrive il tipo di contratto con il cliente, non ancora come si misura. La traduzione in protocollo di benchmark — regime di carico, profili GuideLLM, identificazione della capacità per replica, capacity card — è coperta nel documento dedicato.

> **Riferimento Dielabs:** _"Benchmark Protocol — Validazione empirica del sizing tramite GuideLLM"_. La scelta del benchmark si fa lì, sullo scenario uscito da questo passo.

---

> **_Per un sizing iniziale, da fine-tunare in fase di benchmark, i tre input minimi sono:_**
>
> _— `concorrenza_target` (derivata + headroom, output del passo 3)_ _— contesto max (output del passo 2)_ _— modello (definito al passo 5)_
>
> _Con questi si calcola la KV cache e si dimensiona la vRAM, che guida tutto il resto. La scelta della GPU può emergere dall'iterazione tra passo 6 (sizing), passo 7 (runtime) e passo 8 (hardware)._

---

### 5. Model architecture

**Output:** la decisione sul modello, indipendente da come verrà servito.

- Famiglia / dimensione (dense vs MoE, parametri)
- Precisione / quantizzazione pesi (BF16/FP16 nativo, FP8, INT4)
- Quantizzazione KV cache (FP8/INT8) — decisione separata dai pesi, impatta la memoria dinamica per concorrenza
- Context window nativa
- Tipo di attention (MHA, GQA, MLA) — incide sul footprint del KV cache
- Licenza e provenance — pesi aperti, condizioni commerciali

Perché è un passo dedicato: la scelta del modello vincola ogni decisione downstream (sizing memoria, runtime, hardware, stack). Trattarla come parametro di un passo successivo nasconde il vincolo e produce design incoerenti.

Anti-pattern: "usiamo già Llama 3.1 70B" prima che i passi 1–4 siano scritti. Il modello è un output dei requisiti, non un input.

---

### 6. Sizing

**Output:** numero di GPU per replica, numero di repliche, footprint totale, costo stimato.

**Definizione di replica.** Una replica è una singola istanza completa di serving: modello + GPU assegnate + runtime + TP/PP se previsti + engine process. È l'unità minima di scaling orizzontale, e l'unità su cui si misura `Cr` (capacità per replica) al passo 10.

Iniziamo con un paio di distinzioni utili:

#### Capacity ≠ Performance

|Asse|Domanda|Vincolato da|Metrica|
|---|---|---|---|
|**Capacity**|Quante richieste in volo posso tenere?|Memoria (HBM, KV cache budget)|Concorrenza massima sostenibile|
|**Performance**|Quanto velocemente rispondo per richiesta?|Compute (FLOPS), banda HBM, fabric|TTFT, ITL|

_Le due si chiudono sulla **`concorrenza_target` del passo 3** e sugli **SLO del passo 4**: la concorrenza richiesta deve stare dentro il cap di capacity e mantenere la performance dentro gli SLO. Se sfori una delle due, il sizing non regge._

Confonderle è un errore comune: una GPU può avere abbondante memoria (capacity ok) ma compute insufficiente per il TTFT richiesto, o viceversa avere compute eccellente ma HBM insufficiente per la concorrenza target. Sono due benchmark diversi, due numeri diversi.

#### Modello mentale: vRAM-first, compute-check

Questo framework affronta i due assi in un ordine preciso: **prima 6a (capacity), poi 6b (performance)**. La ragione è strutturale:

- la **vRAM è un vincolo binario** — il sistema parte o non parte
- il **compute è un vincolo continuo** — si misura, si ottimizza con tuning di runtime, si scala

Mettere la vRAM per prima dà un ordine di analisi tracciabile: il sizing della memoria fissa tipo e numero di GPU, e da lì il compute disponibile emerge come conseguenza, da verificare contro gli SLO.

#### Disclaimer logica memory first

_**In alcuni workload il compute è dimensionante quanto la vRAM, e ignorarlo nella prima iterazione produce dimensionamento non corretto:**_

- _Prefill-heavy (RAG con contesti lunghi, document QA): il prefill è compute-bound. Una GPU con vRAM sufficiente ma FLOPS scarsi satura il TTFT anche a bassa concorrenza._
- _MoE: la vRAM nominale dei parametri non riflette il compute reale — solo gli expert attivi consumano FLOPS._
- _Quantizzazione aggressiva (FP8, INT4): cambia il rapporto vRAM/compute in modi non sempre lineari._

_In questi casi l'iterazione tra 6a e 6b è più stretta, e può imporre scelte di GPU che non sarebbero ottimali se si guardasse solo alla vRAM._

---

#### 6a — Capacity sizing

**Output:** memoria per replica, concorrenza massima sostenibile per replica (Cr stimato).

Il calcolo assume che **tutta la KV cache risieda in vRAM**. I meccanismi di offload sono leve successive, fuori dallo scopo di questa guida.

##### Componenti del footprint VRAM

- **Pesi del modello**
- **KV cache** — la quota dominante e variabile, funzione di context length, concorrenza, tipo di attention
- **Activation memory** — buffer intermedi, dipende dal batch size
- **Framework overhead** — CUDA context, allocazioni vLLM, working buffers

##### Formule di sizing VRAM

> ⚠️ **Le formule e le tabelle che seguono sono stime operative, non verità assolute.** Sono sufficienti per scegliere la classe GPU, dimensionare le repliche e impostare conversazioni di sizing con vendor o cliente. **Non sostituiscono il benchmark del passo 10.** I numeri reali variano in funzione di:
>
> - Engine (vLLM, TensorRT-LLM, SGLang) e versione specifica
> - Tipo e implementazione dell'attention (flash attention, xFormers, custom kernel)
> - Schema di quantizzazione (pesi, KV cache, activation)
> - Block size del paged attention e politica di allocazione
> - Padding, frammentazione, allocazioni dinamiche del runtime
>
> _Margine di errore atteso: ±10–15% in condizioni normali. Su MoE: usare i **parametri totali** per la vRAM (tutti i pesi devono stare in HBM), e i **parametri attivi** solo per il sanity check compute al passo 6b._

**1. Pesi del modello**

```
VRAM_pesi = params × bytes_per_param
```

|Precisione|Bytes/param|
|---|---|
|FP16 / BF16|2|
|FP8|1|
|INT8|1|
|INT4 / AWQ / GPTQ 4-bit|0.5|

Esempio: Llama-3 70B in FP16 = 70 × 10⁹ × 2 = ~140 GB.

**2. KV cache per token (per richiesta)**

Formula generale per attention basata su K e V espliciti:

```
KV_per_token = 2 × num_layers × num_kv_heads × head_dim × bytes_per_element
```

- Il fattore `2` è per K e V.
- `num_kv_heads` cambia per tipo di attention:
    - **MHA** (Multi-Head): `num_kv_heads = num_heads`
    - **GQA** (Grouped-Query, es. Llama-3, Mistral): `num_kv_heads < num_heads`, tipicamente 8
    - **MLA** (Multi-head Latent Attention, es. DeepSeek-V2/V3): formula diversa basata su latente compresso, footprint molto inferiore — riferirsi al paper del modello
- `bytes_per_element` segue il dtype del KV cache (FP16 = 2, FP8 = 1, INT8 = 1)

Esempio Llama-3 70B FP16 (80 layer, 8 KV heads, head_dim 128):

```
KV_per_token = 2 × 80 × 8 × 128 × 2 = 327.680 byte ≈ 320 KB/token
```

**3. KV cache totale per richiesta**

```
KV_per_richiesta = KV_per_token × context_length
```

Esempio: Llama-3 70B con context 8K → 320 KB × 8.192 = ~2.6 GB per richiesta.

> **Attenzione al context length da usare nel calcolo.** Si dimensiona sul p95 della distribuzione input+output del passo 2, non sul context massimo del modello. Usare il context massimo nominale (es. 128K per Llama-3) sovradimensiona drasticamente.

**4. Activation memory**

Difficile da prevedere con precisione: dipende da batch size, sequence length, parallelism, e implementazione. Stima indicativa:

```
VRAM_activation ≈ 5–15% dei pesi
```

Per un sizing prudente si usa il 15%. Su modelli quantizzati l'activation resta tipicamente in FP16, quindi pesa di più in proporzione.

**5. Framework overhead**

```
VRAM_overhead ≈ 1–2 GB (CUDA context) + 5% del totale (working buffers vLLM)
```

**6. Memoria utilizzabile sulla GPU**

vLLM non usa il 100% dell'HBM. Il parametro `gpu_memory_utilization` (default 0.9) lascia margine per allocazioni dinamiche e frammentazione:

```
VRAM_utilizzabile = HBM_GPU × gpu_memory_utilization
```

_Valore di riferimento per il sizing: **0.9** (default di entrambi gli engine, anche se vLLM è recentemente passato a 0.92). In produzione si scende spesso a **0.8–0.85** per modelli grandi, configurazioni multi-nodo o quando si osservano OOM occasionali._

**7. Concorrenza massima per replica (Cr stimato)**

Mettendo tutto insieme:

```
VRAM_disponibile_per_KV = VRAM_utilizzabile − VRAM_pesi − VRAM_activation − VRAM_overhead

Cr_stimato = VRAM_disponibile_per_KV / KV_per_richiesta
```

**Esempio completo: Llama-3 70B FP16 su H100 80GB, context p95 = 8K**

```
VRAM_utilizzabile     = 80 × 0.9         = 72 GB
VRAM_pesi             = 70 × 2           = 140 GB  ← non sta su una sola GPU
```

Il modello non entra: serve TP=2 (passo 7). Su 2× H100:

```
VRAM_utilizzabile     = 2 × 72           = 144 GB
VRAM_pesi (TP=2)      = 140 / 2 × 2      = 140 GB totali, 70 per GPU
VRAM_activation       ≈ 140 × 0.15       ≈ 21 GB totali, ~10 per GPU
VRAM_overhead         ≈ 2 + 5% × 144     ≈ 9 GB totali
VRAM_disponibile_KV   = 144 − 140 − 21 − 9 = ~−26 GB    ← non basta
```

Servono 4× H100 con TP=4, oppure 2× H200 (141 GB ciascuna), oppure quantizzazione FP8 per dimezzare i pesi. Questo è il tipo di iterazione che il passo 6 forza prima di scegliere l'hardware al passo 8.

##### Leve runtime che modificano la formula

Le scelte al passo 7 cambiano i numeri sopra:

- **Weight quantization** (FP8, INT8, INT4/AWQ/GPTQ) → riduce `VRAM_pesi` in proporzione (FP8 = ½, INT4 = ¼)
- **KV cache quantization** (FP8/INT8) → dimezza `KV_per_token`
- **Prefix caching** → riduce il KV effettivo per richiesta in proporzione al tasso di hit del prefisso (workload-dipendente)
- **KV offload** (CPU/NVMe) → sposta parte del KV fuori dall'HBM, alza la concorrenza max ma incide sulla performance (passo 6b)
- **Tensor parallelism** → divide pesi e KV cache per il grado di TP, a costo di banda inter-GPU

Si valuta se applicare queste leve quando `Cr_stimato` è inferiore alla `concorrenza_target` del passo 3.

> _**Weight quantization e KV cache quantization sono leve indipendenti.**_ Quantizzare i pesi (es. con `--quantization fp8`) non quantizza automaticamente il KV cache, che resta nella precisione nativa del modello finché non si attiva esplicitamente `--kv-cache-dtype`. Conseguenza pratica: per dimezzare entrambe le voci di vRAM servono due decisioni distinte, non una. È un errore di sizing comune assumere che la prima implichi la seconda.

##### Crescita futura

_just a little reminder for you_

##### Modelli di riferimento (cheat-sheet)

Parametri architetturali dei modelli più comuni in produzione, con KV cache per token in FP16. Tutti usano GQA tranne dove indicato.

> **Stima operativa.** I valori in tabella sono calcolati con la formula del KV cache standard. I numeri reali sotto carico vLLM possono differire del ±10–15% per via di paged attention block size, padding, e overhead di allocazione. Usare per il design, validare al passo 10.

|Modello|Layers|KV heads|head_dim|KV/token (FP16)|Pesi FP16|Pesi FP8|Pesi INT4|
|---|---|---|---|---|---|---|---|
|Llama-3 8B|32|8|128|128 KB|16 GB|8 GB|4 GB|
|Llama-3 70B|80|8|128|320 KB|140 GB|70 GB|35 GB|
|Mistral 7B v0.3|32|8|128|128 KB|14 GB|7 GB|3.5 GB|
|Qwen2.5 7B|28|4|128|56 KB|15 GB|7.5 GB|3.8 GB|
|Qwen2.5 32B|64|8|128|256 KB|64 GB|32 GB|16 GB|
|Qwen2.5 72B|80|8|128|320 KB|144 GB|72 GB|36 GB|

**KV cache per richiesta a context tipici (FP16):**

|Modello|4K context|8K context|32K context|
|---|---|---|---|
|Llama-3 8B / Mistral 7B|0.5 GB|1.0 GB|4.0 GB|
|Qwen2.5 7B|0.2 GB|0.5 GB|1.8 GB|
|Llama-3 70B / Qwen2.5 72B|1.3 GB|2.6 GB|10.5 GB|
|Qwen2.5 32B|1.0 GB|2.0 GB|8.4 GB|

**Esempio rapido: Llama-3 8B FP16 su L40S 48GB, context p95 = 8K**

```
VRAM_utilizzabile     = 48 × 0.9       = 43.2 GB
VRAM_pesi             = 16 GB
VRAM_activation       ≈ 16 × 0.15      ≈ 2.4 GB
VRAM_overhead         ≈ 2 + 5% × 43.2  ≈ 4 GB
VRAM_disponibile_KV   = 43.2 − 16 − 2.4 − 4 ≈ 21 GB
KV_per_richiesta      = 1 GB (8K context)
Cr_stimato            = 21 / 1         ≈ 21 richieste
```

Una L40S regge ~20 richieste concorrenti su Llama-3 8B a 8K. Per scalare oltre: più repliche o GPU con più HBM.

##### Esito

_Il `Cr_stimato` per replica è il cap di capacity. Se è inferiore alla `concorrenza_target` del passo 3, le opzioni sono: più repliche (sizing aggregato), GPU con più HBM (passo 8), leve ottimizzazione runtime al passo 7, modello più piccolo o più quantizzato (passo 5)._

#### 6b — Performance sizing

**Output:** verifica che la GPU scelta per capacity rispetti gli SLO di latenza alla `concorrenza_target`.

Un concetto importante prima di iniziare:

**MFU = Model FLOPS Utilization.** È la percentuale dei FLOPS teorici di picco di una GPU che il modello riesce effettivamente a usare durante l'esecuzione.

```
MFU = FLOPS_effettivi / FLOPS_picco_GPU
```

Non si arriva mai al 100% perché il picco da datasheet assume condizioni ideali (operazioni matriciali perfettamente ottimizzate, niente overhead di trasferimento dati, niente sincronizzazione). Nella realtà subentrano accessi alla memoria HBM, comunicazione tra GPU, kernel launch, etc.

La scelta GPU che soddisfa la capacity potrebbe non soddisfare gli SLO di performance. Tre verifiche da fare:

- **TTFT (compute-bound)** — FLOPS richiesti per il prefill al p95 della distribuzione input → tempo di prefill stimato vs SLO TTFT
- **ITL (banda-bound)** — banda HBM richiesta per il decode al batch size target → tempo per token vs SLO ITL
- **Fabric** — banda inter-nodo richiesta da TP/PP multi-nodo (se previsto al passo 7), banda KV transport (se disaggregato), banda dello storage tier (se offload KV previsto)

Quando capacity e performance richiedono GPU diverse, **vince il vincolo dominante:** tipicamente la memoria su modelli grandi a context lungo, il compute su modelli piccoli ad alta concorrenza con SLO TTFT stretti.

> **Reminder sulla coerenza E2E.** Prima di iniziare la verifica TTFT/ITL, rileggere la formula `E2E ≈ TTFT + ITL × output_p95` introdotta in 4.1. Se il TTFT target da solo già consuma gran parte del budget E2E, il margine residuo per il decode è risicato e il sizing di 6b deve mirare a un ITL più aggressivo del riferimento generico — oppure la conversazione torna al passo 4 per rinegoziare gli SLO.

##### Sanity check rapido sul TTFT (prefill compute-bound)

Per stimare se la GPU scelta a 6a regge il prefill al p95 entro lo SLO TTFT, si confrontano FLOPS richiesti e FLOPS effettivi disponibili.

```
FLOPS richiesti per prefill ≈ 2 × N_params × input_tokens × concorrenza
tempo prefill ≈ FLOPS richiesti / FLOPS GPU effettivi
FLOPS GPU effettivi ≈ peak FLOPS × MFU
```

Il **prefill è il caso favorevole per l'MFU** perché processa molti token in un singolo forward pass: in questa fase un MFU realistico è **0.4–0.6**. Nel decode invece l'MFU crolla drasticamente (può scendere sotto il 5%), perché la generazione sequenziale è memory-bandwidth-bound, non compute-bound — ma per il decode il vincolo dimensionante non è il compute, è la banda HBM (verifica ITL).

**Riferimento: H100 SXM da datasheet NVIDIA**

|Precisione|TFLOPS dense|TFLOPS con sparsity 2:4|
|---|---|---|
|BF16 / FP16 Tensor Core|~990|1.979|
|FP8 Tensor Core|~1.979|3.958|

I valori con sparsity si applicano solo se il modello è stato preparato per sfruttarla — caso raro in inference standard. Per il sanity check si usa il valore dense.

**Esempio: Llama-3 70B BF16, input p95 = 4000 tok, concorrenza = 8, H100 SXM**

```
FLOPS richiesti     ≈ 2 × 70e9 × 4000 × 8     ≈ 4,5 PFLOPS
FLOPS GPU effettivi ≈ 990e12 × 0,5            ≈ 495 TFLOPS
tempo prefill       ≈ 4,5e15 / 495e12         ≈ 9,1 s
```

Se lo SLO di TTFT è 2 s, una sola H100 in BF16 non basta per il prefill aggregato a quella concorrenza: serve parallelismo (TP), più repliche, oppure quantizzazione FP8 che raddoppia il throughput compute. Se lo SLO è 30 s (batch processing), c'è margine.

> **Calcolo a spanne, non sizing.** Serve a capire se il compute è in zona pericolosa già al passo 6, evitando di arrivare al benchmark del passo 10 con sorprese strutturali. Il benchmark resta l'unica fonte di verità.

#### Sizing aggregato

Una volta fissate GPU e concorrenza per replica, il sizing aggregato chiude i conti:

- **Repliche necessarie** = `concorrenza_target` del passo 3 / Cr stimato per replica
- **Ridondanza N+1 (o N+2)** — moltiplicatore esterno per il dominio di failure. Non cambia la singola replica, aggiunge un'unità intera per failover.
- **GPU per replica × conteggio repliche** → conteggio GPU totale

> **Nota.** Il sizing a questo punto è una stima basata su scelte architetturali e benchmark di riferimento, non sul comportamento del sistema completo sotto carico reale. Il passo 10 valida empiricamente sia il cap di capacity sia i numeri di performance, e può richiedere aggiustamenti se la realtà si discosta.

---

### 7. Runtime architecture

**Output:** lo strato software che esegue il modello su una replica, definendone la distribuzione dei parametri e le politiche di scheduling delle richieste.

- **Parallelismo / Sharding** — TP, PP, EP, DP e loro combinazioni
- **KV cache strategy** — paged attention, prefix caching, offload (CPU/NVMe), quantizzazione del KV
- **Architettura aggregata vs disaggregata** — _vedi sotto_
- **Speculative decoding** — sì/no, scelta del draft model
- **Regime di scheduling** — continuous batching, chunked prefill, priority lanes

_**Possibili co-design con i passi 5 e 6**_

- _Il modello sta su una sola GPU? → se no, TP/PP_
- _MoE? → considera EP_
- _Context lungo + concorrenza alta? → pressione sul KV cache → offload o disaggregazione_
- _SLO TTFT stretto? → worker di prefill dedicati_

#### Il bivio architettura aggregata vs disaggregata

L'inferenza disaggregata non è un'ottimizzazione del serving stack. È una decisione architetturale a livello di runtime che separa un singolo servizio di inferenza in due servizi accoppiati: un **servizio di compute stateless** (prefill) e un **servizio di memoria stateful** (decode + KV cache).

**Aggregato** — un servizio, un processo, un dominio di failure. Operazioni più semplici, overhead di latenza più basso, TCO più basso a scala piccola/media. Il punto di partenza di default.

**Disaggregato** — due servizi con trasferimento del KV cache sull'interconnessione (NVLink, InfiniBand, RoCEv2, NIXL). Permette scaling indipendente di prefill e decode, fleet GPU eterogenei, e migliore utilizzo a scala. Introduce nuovi failure mode:

- **Rate-matching mismatch** tra pool prefill e decode — il vero rischio operativo
- **Trasporto del KV** come nuovo critical path (latenza, banda, congestione)
- **Due domini di failure** invece di uno, ciascuno con scaling e osservabilità propri

Si sceglie il disaggregato quando la scala, l'eterogeneità del fleet o lo squilibrio del workload giustificano il costo operativo. Altrimenti l'aggregato vince su TCO e semplicità.

> Il disaggregato merita una trattazione dedicata. Riferimento Dielabs _"Disaggregated inference: prefill/decode separation, KV transport, and operational tradeoffs"_ per la rottura architetturale completa, i dettagli del trasporto NIXL e l'analisi del rate-matching.

_**In un design greenfield, il runtime dovrebbe guidare la scelta hardware:**_ parallelism e KV strategy determinano i requisiti del fabric. TP multi-nodo richiede NVLink esteso o IB ad alta banda; il disaggregato richiede RDMA per il trasferimento del KV. Scegliere l'hardware prima del runtime in greenfield significa scoprire poi che il fabric non regge il runtime desiderato.

_**In contesti enterprise on-prem, l'hardware è spesso un vincolo iniziale:**_ budget approvato, rack disponibili, power budget, contratti vendor, disponibilità GPU sul mercato, parco esistente da riutilizzare. In questi casi la sequenza si inverte: il runtime va adattato e validato contro l'hardware disponibile. Il framework resta lo stesso, cambia il punto di ingresso del vincolo. La domanda da farsi esplicitamente al cliente al passo 1 è: _l'hardware è una variabile o un vincolo?_ — la risposta determina come si percorre il loop 6-8.

---

### 8. Hardware & fabric architecture

**Output:** il substrato fisico su cui atterra il runtime.

**Compute**

- Classe GPU (H100, H200, B200, MI300X, L40S, acceleratori non-NVIDIA)
- Memoria per GPU — fissa il limite superiore di KV cache + pesi
- GPU per nodo — determina le opzioni di parallelism intra-nodo

**Fabric**

- **Intra-nodo:** generazione NVLink / NVSwitch, topologia PCIe
- **Inter-nodo:** InfiniBand vs RoCEv2, velocità del link, rapporto di oversubscription
- **Fabric di storage:** due workload distinti sullo stesso layer fisico
    - _Caricamento pesi del modello_ — cold start, scaling delle repliche, swap di modello. Bandwidth-bound, lettura singola, accesso sequenziale.
    - _Tier di offload del KV cache_ — RAM CPU, NVMe locale, o pool remoti. Latency-bound, lettura/scrittura, accesso casuale. Critical path per workload long-context e ad alta concorrenza.

**Perché il fabric è una preoccupazione propria:** le scelte di parallelism del passo 7 sono praticabili solo se il fabric le supporta. TP tra nodi senza NVLink è una trappola di performance. Trasferimento KV disaggregato senza RDMA è una trappola di latenza.

**Lo storage fabric non è solo per i pesi.** Quando il runtime (passo 7) sceglie l'offload del KV cache — su RAM CPU, NVMe locale, o pool remoto — lo storage fabric diventa parte del critical path dell'inferenza, non solo una preoccupazione di startup. Latenza e banda del tier di offload incidono direttamente sul TTFT per cache hit e sul throughput di decode quando il working set eccede la memoria GPU.

**KV cache come sistema di memoria a tier.** L'inferenza moderna tratta la memoria come una gerarchia: HBM (tier 1) → RAM CPU (tier 2) → NVMe locale (tier 3) → pool remoto (tier 4). Ogni tier ha caratteristiche distinte di latenza, banda e costo. Il passo 7 decide _quali tier usare_; il passo 8 decide _se il fabric li sostiene_.

> Storage fabric e offload KV meritano una trattazione dedicata. Riferimento Dielabs _"KV cache as a tiered memory system: from HBM to remote pools"_ per la gerarchia completa, i budget di latenza per tier e i criteri di selezione del tier in fase di design.

**Fleet eterogeneo:** workload diversi beneficiano di GPU diverse (decode-heavy → memory-bound, prefill-heavy → compute-bound). Il _bin packing_ — assegnare ogni workload alla classe di GPU che lo serve in modo più efficiente, riempiendo le GPU sul loro punto forte, usando a volte anche quelle "in dismissione" — su un fleet misto è una leva di TCO, non una complicazione da evitare.

---

### 9. Serving stack

**Output:** la piattaforma di serving (orchestrazione, routing) sopra il runtime, con le leve di configurazione e tuning per **adattare il sistema al workload** e portarlo verso gli SLO.

#### Le tre famiglie di ottimizzazione

Le leve del serving stack si raggruppano in tre famiglie, ciascuna che attacca un bottleneck specifico. Il razionale di ognuna spiega _perché_ funziona; i valori efficaci dei parametri si trovano sperimentalmente con il benchmark del passo 10.

|Famiglia|Target|Tecniche principali|Parametri vLLM|
|---|---|---|---|
|**Opt throughput / prefill**|TTFT, throughput|Chunked prefill, prefix caching, batch grandi|`max-num-batched-tokens` (alto), `enable-prefix-caching`|
|**Opt latenza / decode**|ITL, tempo per token|Continuous batching, speculative decoding, batch piccoli|`max-num-seqs` (basso), flag speculative, `max-num-batched-tokens` (basso)|
|**Opt VRAM / capacity**|Concorrenza, context length|Paged attention, KV cache quantization, weight quantization, KV offload|`gpu-memory-utilization`, `kv-cache-dtype`, `quantization`|

##### Opt throughput / prefill — perché funziona

Il prefill è la fase più costosa in compute: deve calcolare K e V per _tutti_ i token del prompt prima che inizi il decode. Le leve di questa famiglia mirano a saturare meglio i tensor core e a ridurre il lavoro ridondante.

- **Chunked prefill** spezza prompt lunghi in chunk e li processa intercalati con il decode di altre richieste. Senza chunked prefill un prompt da 32K blocca completamente il decode di tutti gli altri utenti finché il prefill non finisce. Con chunked prefill, il decode procede in parallelo. → migliora TTFT sotto carico misto e throughput aggregato.
- **Prefix caching** riconosce che molte richieste condividono il prefisso (system prompt, few-shot examples, context RAG comuni) e riusa il KV cache già calcolato. → riduce il lavoro di prefill in proporzione al tasso di cache hit. Su workload con system prompt lungo e ripetuto può tagliare il TTFT del 50%+.
- **Batch più grandi nel prefill** (`max-num-batched-tokens` alto) saturano meglio i tensor core, che sono compute-bound. → migliore throughput, a costo di TTFT più alto sui prompt singoli.

##### Opt latenza / decode — perché funziona

Il decode è memory-bandwidth-bound: per ogni token generato, la GPU rilegge tutti i pesi del modello e tutto il KV cache della richiesta. Le leve di questa famiglia mirano a generare più token per ogni passata di lettura, o a ridurre i token da generare.

- **Continuous batching** raggruppa nello stesso forward pass token in decode di richieste diverse. Una passata di lettura dei pesi serve N richieste invece di una. → migliora throughput senza penalizzare l'ITL della singola richiesta. È la base di vLLM, attivo di default.
- **Speculative decoding** usa un modello piccolo (draft model) per generare candidati che il modello grande verifica in batch. Più token validati per ogni passata del modello grande. → riduce ITL su workload predicibili (codice, formati strutturati). Non sempre conviene: il draft deve essere abbastanza accurato, altrimenti l'overhead di verifica annulla il guadagno.
- **Batch piccoli nel decode** (`max-num-seqs` basso) riducono la contesa di banda HBM tra le richieste in volo perché le diminuisce. → migliora ITL p99 al costo del throughput aggregato. _Attenzione: se la concorrenza in arrivo supera il numero di richieste ammesse in parallelo, le richieste in eccesso vanno in coda e il TTFT sale — l'ITL della singola richiesta rimane migliore, ma l'esperienza serving peggiora. Conviene solo se la concorrenza sta sotto quel limite, o se ci sono abbastanza repliche per assorbire l'overflow._

##### Opt VRAM / capacity — perché funziona

La VRAM è il vincolo dominante della concorrenza (vedi passo 6a). Le leve di questa famiglia liberano memoria per ospitare più richieste in volo, allungare il context, o entrare in GPU più piccole.

- **Paged attention** alloca il KV cache a blocchi di dimensione fissa invece che in spazi contigui per richiesta. → elimina la frammentazione e il padding, recuperando tipicamente il 20–30% della VRAM rispetto all'allocazione contigua. È la base di vLLM, attivo di default.
- **KV cache quantization** (FP8, INT8) riduce il footprint del KV cache di 2x rispetto al baseline a 16 bit (BF16 o FP16). → raddoppia approssimativamente la concorrenza max per replica. Costo: piccola perdita di accuracy, da verificare sul tuo workload. È una leva indipendente dalla weight quantization (vedi passo 6a). _Nota: con FP8 KV cache, il calcolo dell'attention avviene nativamente in FP8 solo con Flash Attention 3 (guadagno sia in capacity sia in compute). Sugli altri backend il KV cache viene tipicamente dequantizzato a BF16/FP16 prima del calcolo, quindi il guadagno principale è in capacity, non in compute._
- **Weight quantization** (FP8, INT8, INT4/AWQ/GPTQ) riduce il footprint dei pesi. → libera VRAM per KV cache e activation, oppure permette di entrare in GPU più piccole. Costo: perdita di accuracy variabile per metodo e modello.
- **KV offload** (CPU RAM, NVMe) sposta parte del KV cache fuori dall'HBM. → alza il cap di concorrenza, ma introduce latenza di trasferimento sul critical path. Conviene quando il workload tollera TTFT più alti per cache hit (es. RAG con prefisso lungo cacheable).

#### Componenti dello stack oltre l'engine

- **Engine** — vLLM, TensorRT-LLM, SGLang, llama.cpp
- **Layer API** — endpoint OpenAI-compatible, auth, rate limiting
- **Router / gateway** — selezione replica, routing modello, fallback
- **Autoscaler** — KEDA, HPA custom su metriche engine
- **Osservabilità** — Prometheus, Grafana, tracing, attribuzione costo per tenant

#### Workflow di tuning

1. **Parti dai default** di vLLM. Sono ragionevoli per la maggior parte dei workload.
2. **Esegui il benchmark del passo 10** e identifica quale SLO non regge.
3. **Modifica una leva alla volta** della famiglia corrispondente al bottleneck.
4. **Rifai il benchmark** e confronta. Se è migliorato, tieni; altrimenti rollback.
5. **Itera** finché tutti gli SLO reggono o le leve sono esaurite (a quel punto si torna al passo 7 o 8).

Modificare più leve insieme rende impossibile capire quale ha avuto effetto. Una alla volta è la regola.

#### Mini-glossario parametri vLLM

I parametri che si toccano più spesso durante il workflow di tuning. Per la lista completa, riferirsi alla documentazione ufficiale di vLLM.

|Parametro|Effetto|
|---|---|
|`max-num-batched-tokens`|Tetto totale di token (prefill + decode) processati in un forward pass. Alto → throughput e TTFT migliori. Basso → ITL migliore.|
|`max-num-seqs`|Tetto di richieste in volo simultaneamente. Cap sulla concorrenza per replica indipendente dalla VRAM.|
|`gpu-memory-utilization`|Frazione di HBM utilizzabile da vLLM (default 0.9). Alzarlo aumenta lo spazio disponibile per il KV cache, abbassarlo lascia più margine per allocazioni dinamiche e overhead.|
|`enable-prefix-caching`|Attiva il riuso del KV cache per prefissi condivisi tra richieste.|
|`kv-cache-dtype`|Precisione del KV cache (`auto`, `fp8`). Default `auto` = stessa precisione del modello (BF16/FP16). FP8 dimezza il footprint del KV.|
|`quantization`|Schema di quantizzazione dei pesi (`awq`, `gptq`, `fp8`). Riduce VRAM occupata dai pesi.|
|`enable-chunked-prefill`|Spezza prefill lunghi in chunk intercalati con il decode. Default attivo in vLLM V1.|
|`max-model-len`|Context length massimo accettato per richiesta. Allinearlo al p95 reale del workload aumenta la concorrenza effettiva: lo scheduler ammette più richieste in parallelo perché ognuna ha un cap di crescita inferiore. Non libera VRAM in senso letterale (paged attention alloca blocchi dinamicamente), ma riduce le preemption sotto carico.|

---

### 10. Benchmark & baseline

**Output:** validazione empirica del sizing, envelope operativo documentato, eventuali aggiustamenti del sizing del passo 6.

I passi precedenti producono un sizing basato su stime architetturali e formule di riferimento. Il passo 10 lo mette sotto carico e verifica se regge.

#### Cosa il passo deve confermare

Indipendentemente dallo scenario uscito da 4.2, il benchmark deve produrre due conferme di business:

- **Il sizing di capacity regge** — la `concorrenza_target` del passo 3 sta dentro la capacità misurata per replica × numero di repliche pianificato.
-
- **Gli SLO di latenza reggono** — TTFT, ITL, E2E p95 misurati sotto il carico target sono dentro i target del passo 4.

#### Due cose che il benchmark non dice da solo

**La coda conta.** TTFT e ITL misurati dentro l'engine ignorano il tempo passato in coda sotto carico. L'utente percepisce l'end-to-end. È un errore comune in benchmark di prima implementazione: l'E2E in produzione peggiora invece di migliorare perché la coda non era nel grafico. Il benchmark deve riportare l'E2E inclusiva della coda, non solo le metriche engine-side.

**Costo della coda come segnale diagnostico.** L'E2E p95 misurato va confrontato con `E2E_atteso = TTFT + ITL × output_p95` calcolato in 4.1. Il _costo della coda_ è la differenza tra i due: `costo_coda = E2E_misurato − E2E_atteso`. Misura il tempo che le richieste passano in attesa di essere servite sotto carico — tempo che la formula non vede. Due casi di sforamento SLO:

- **Costo della coda alto** → la singola richiesta sarebbe dentro il budget, è la concorrenza che accumula richieste in attesa. Bottleneck di capacità → passo 11 con scaling o tuning.
- **Costo della coda basso** → l'E2E sfora anche senza coda significativa; la singola richiesta è già troppo lenta a coda vuota. Bottleneck di performance per richiesta → riapre il passo 6b.

| Costo della coda       | Significato                                   | Dove sta il problema                              | Dove torni                    |
| ---------------------- | --------------------------------------------- | ------------------------------------------------- | ----------------------------- |
| **Alto**               | La coda divora il budget                      | Capacità sotto carico                             | Passo 11 (scaling/tuning)     |
| **Basso** ma E2E sfora | La singola richiesta è già lenta a coda vuota | Design per richiesta (GPU, parallelismo, modello) | Passo 6b (performance sizing) |

#### Esecuzione

Il protocollo operativo — scelta del benchmark in base allo scenario, profili di carico, identificazione della capacità per replica, metriche da raccogliere, capacity card — è coperto nel documento dedicato.

> **Riferimento Dielabs: _"Benchmark Protocol — Validazione empirica del sizing tramite GuideLLM"_. Documento standalone con prerequisiti dichiarati.**

#### Esito

Il sizing del passo 6 può uscire confermato, oppure richiedere aggiustamenti (numero di repliche, classe GPU, leve di runtime). Gli aggiustamenti tipici e la decisione di scaling sono trattati al passo 11.

---

### 11. Scaling decision / iterazione

**Output:** scaling consapevole, basato sul bottleneck identificato.

Se la baseline non chiude rispetto ai target del passo 4, non si scala con repliche aggiuntive senza prima capire perché. Le repliche moltiplicano: un'inefficienza nella baseline la paghi N volte.

#### Workflow

###### 1. Identifica il bottleneck dominante** dalle metriche, non dall'intuizione.

| Bottleneck                    | Segnale                                          | Famiglia      |
| ----------------------------- | ------------------------------------------------ | ------------- |
| KV cache pressure             | `vllm:gpu_cache_usage_perc` > 90%, preemption    | Memoria       |
| Prefill compute-bound         | GPU utilization alta, TTFT alto su prompt lunghi | Compute       |
| Decode memory-bandwidth-bound | Banda HBM satura, ITL alto                       | Memoria/banda |
| Queue-bound                   | E2E >> latenza engine                            | Capacità      |
| Cost over target              | €/Mtok > target con SLO rispettati               | Design        |

Senza osservabilità (metriche vLLM su Prometheus) non vedi il bottleneck, vedi solo il sintomo aggregato.

###### 2. Scegli la leva in base a **costo e complessità**, non per default.
###### L'ordine sotto è una buona prassi: dal più economico e meno invasivo al più radicale. Eccezione: le repliche orizzontali sono la leva più semplice da modellare in un BoM, ma vanno _dopo_ tuning e runtime, non _prima_, per evitare di moltiplicare un'inefficienza.

- **Tuning del serving stack** (parametri vLLM, prefix caching, chunked prefill, quantizzazione KV) → leva quasi gratuita. Da provare sempre per prima. Spesso recupera il 20–40% di throughput. Torna al passo 9.

- **Modifiche al runtime** (parallelism, disaggregazione P/D, speculative decoding) → potente ma operativamente complessa. Giustificata quando il bottleneck è strutturale. Torna al passo 7.

- **Modifiche all'hardware** (più HBM, fabric più veloce, GPU diverse) → costoso ma necessario se il fabric è il limite. Torna al passo 8.

- **Repliche orizzontali** → semplici da modellare in un BoM, ma inefficienti in €/token se mascherano un'inefficienza upstream. Torna al passo 10 per ri-dimensionare.

- **Cambio modello** (size più piccolo, quantizzazione più aggressiva) → leva radicale, riapre tutto il design. Torna al passo 5.

###### 3. Se scali con repliche, tieni conto del pattern di traffico del passo 3.1.** (**advanced**)
Su traffico costante, repliche statiche bastano. Su traffico bursty, l'autoscaling è necessario?repliche statiche dimensionate sul picco fanno pagare l'idle, dimensionate sulla media violano gli SLO. L'autoscaling va configurato sulle metriche giuste — `vllm:num_requests_running + waiting` e `vllm:gpu_cache_usage_perc` — non su CPU o GPU utilization, che sono i default di HPA ma non riflettono la saturazione reale di una replica di inferenza.

#### Il framing

> _Identifica il bottleneck prima di scegliere la leva. Le repliche sono una scelta consapevole "finale", non un default._

Questo separa un capacity planner ("aggiungi nodi") da un architetto ("rate-match prima, poi decidi se scalare"). Le repliche sono la leva più semplice da modellare in un BoM, ma non sempre la più efficiente in €/token. Mascherare un sintomo (latenza alta sotto carico) scalando le repliche fa sparire il sintomo ma porta l'inefficienza configurazionale moltiplicata in produzione.

**Anti-pattern:** "ottimizza tutto prima di scalare". L'ottimizzazione perfetta non esiste, e cercarla diventa un alibi per non scalare mai. La regola pratica è: applicare le leve a basso costo che agiscono sul bottleneck identificato, e scalare con repliche solo quando le leve rimanenti hanno un costo/complessità comparabile o superiore.

---

### Principi guida

Il sizing non è un numero singolo. È un envelope (uno o più, secondo lo scenario) più un set di margini di sicurezza.

1. **Worst case, non media** — i picchi rompono gli SLO, le medie li nascondono
2. **Headroom di provisioning esplicito** — un singolo margine sopra la concorrenza p95, dimensionato sul massimo tra crescita prevista e assorbimento di bursts
3. **Dominio di failure N+1** — un nodo giù ≠ violazione SLO (N+2 per workload critici)
4. **Capacity ≠ performance** — due assi distinti del sizing (6a vs 6b), che possono richiedere uno o due benchmark al passo 10 a seconda del profilo di SLO
5. **TCO, non throughput** — €/1M token è la metrica di business, catturata nell'envelope throughput-first
6. **Osservabilità dal giorno 1** — se non puoi misurarlo in prod, non puoi gestirlo
7. **Assumi che le assunzioni siano sbagliate** — il primo mese in prod è misurazione, non steady state
8. **Fleet eterogeneo come leva di TCO, non complicazione da evitare** — workload diversi beneficiano di GPU diverse, il bin-packing è una scelta architetturale con benefici
9. **Pianifica l'uscita, non solo l'ingresso** — i modelli cambiano, il workload raddoppia, gli engine evolvono
10. **Attenzione alle giunzioni** — la maggior parte dei failure vive tra i layer (runtime↔fabric, runtime↔stack, HBM↔offload tier), non dentro di essi

---

### Perché questo framework

La maggior parte dei capacity plan fallisce perché salta i passi 1–4 e parte dal passo 9. Tunare i flag dell'engine senza un workload profile è tirare a indovinare. Tunare senza SLO è ottimizzare senza un target. Ottimizzare la latenza interna all'engine senza misurare l'end-to-end è risolvere il problema sbagliato. Aggiungere repliche senza identificare il bottleneck è moltiplicare l'inefficienza.

Il framework forza la conversazione nell'ordine giusto: **business → forma del carico → volume del carico → contratto → modello → sizing → runtime → hardware → stack → benchmark → scaling consapevole**.

Il framework è opinionato su cinque punti:

1. **Workload profile e traffic profile sono due input distinti.** La forma della richiesta (distribuzione) e il volume nel tempo (serie temporale) si stimano con tecniche diverse e si negoziano con il cliente in conversazioni diverse. Confonderli produce sizing su numeri medi.
2. **Runtime, hardware e serving stack sono tre preoccupazioni distinte.** Confonderle è la fonte più comune di design di inferenza incoerenti.
3. **Il sizing è una stima validata empiricamente.** Il passo 6 produce numeri basati su scelte architetturali e benchmark di riferimento; il passo 10 li valida sotto carico reale e può richiedere aggiustamenti.
4. **Il runtime guida l'hardware in greenfield, l'hardware vincola il runtime in enterprise.** Esplicitare al passo 1 se l'hardware è variabile o vincolo determina come si percorre il loop 6-8.
5. **Lo scaling è una decisione, non un default.** Il passo 11 chiude il ciclo identificando il bottleneck e scegliendo la leva con consapevolezza di costo e complessità.

---

_Framework sviluppato in [Dielabs](https://github.com/) — laboratorio personale di ricerca su infrastruttura di inferenza LLM. Feedback benvenuti._
