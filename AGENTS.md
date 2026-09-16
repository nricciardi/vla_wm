# Istruzioni editoriali

Questa directory contiene una raccolta strutturata di appunti e conoscenze su Vision-Language-Action model e World Model. Il risultato deve poter essere letto sia come un libro tecnico sia come materiale da presentare a tutor accademici.

## Stile

- Scrivere in italiano con un registro formale ma non pesante.
- Preferire una trattazione discorsiva e collegare logicamente definizioni, motivazioni e conseguenze.
- Suddividere il testo in paragrafi brevi, ciascuno dedicato a un passaggio logico riconoscibile, e separare sempre i paragrafi con una riga vuota. Evitare muri di testo anche quando gli argomenti sono strettamente collegati.
- Evidenziare in **grassetto** i concetti chiave e le formulazioni che permettono di ricostruire rapidamente il contenuto della sezione, senza abusare dell'enfasi.
- Essere dettagliati senza dare per scontati i concetti necessari alla comprensione.
- Introdurre i formalismi quando chiariscono il testo, evitando passaggi matematici non motivati o inutilmente pesanti.
- Definire ogni simbolo quando viene introdotto e spiegare in prosa il significato delle equazioni.
- Usare i termini inglesi consolidati quando sono quelli prevalenti nella letteratura, spiegandoli alla prima occorrenza se necessario.
- Evitare elenchi di affermazioni isolate quando lo stesso contenuto può essere esposto più chiaramente in forma discorsiva.
- Non usare separatori orizzontali `---` per dividere sezioni o passaggi del testo; affidarsi alla gerarchia degli header e alla separazione in paragrafi.

## Struttura dei contenuti

- Organizzare gli argomenti in file e sottodirectory coerenti. Ogni capitolo importante o corposo deve avere una sottodirectory dedicata con il testo completo; questo vale anche per l'analisi di un singolo paper e per prerequisiti estesi come il controllo robotico. Una sezione realmente breve può rimanere nel README che la introduce.
- Usare i README di livello superiore come introduzioni e indici ragionati. Il testo completo non deve essere duplicato nel README principale.
- Per ogni approfondimento, mantenere nel README principale un summary autosufficiente che permetta di comprenderne il ruolo senza aprire immediatamente il file dedicato.
- Il summary di un paper deve chiarire, in forma discorsiva e concisa, l'intuizione centrale, i dataset utilizzati, il robot o gli embodiment coinvolti, la novelty e le limitazioni. Se una voce non è applicabile, come il robot per un argomento teorico, non introdurre campi artificiali o informazioni prive di significato.
- Per ogni modello, presentare di norma novelty e limiti in due sottosezioni distinte di quarto livello, intitolate `#### Novelty` e `#### Limiti`. Sviluppare ciascuna sottosezione in uno o più paragrafi autonomi, senza accorpare novelty e limiti nello stesso paragrafo.
- Concludere il summary con un collegamento naturale all'approfondimento nella relativa sottodirectory.
- Inserire i link agli approfondimenti nel corpo del testo, non nel titolo della sezione.
- Non anteporre numeri ai titoli e ai sottotitoli. I numeri sono ammessi quando costituiscono realmente un elenco ordinato.
- Non concludere i file con sezioni denominate “Sintesi”, “Conclusioni riepilogative” o “Letture essenziali”. Integrare nel testo i collegamenti concettuali e inserire eventuali riferimenti vicino alle affermazioni che supportano.
- Evitare ripetizioni tra il README introduttivo e i file di approfondimento: il primo deve orientare il lettore, i secondi devono sviluppare l'argomento.

## Notazione

- Delimitare sempre il LaTeX inline con `$...$` e quello su riga separata con `$$...$$`; non usare `\(...\)` o `\[...\]`.
- Usare $o_t$ per l'osservazione, $a_t$ per l'azione e $s_t$ per lo stato, salvo esigenze motivate e dichiarate nel testo.
- Usare sempre $q$ per rappresentare l'istruzione linguistica, incluse le sezioni dedicate a RT-1 e RT-2.
- Non sostituire con $q$ gli indici matematici privi di significato linguistico, come l'indice di un esempio o di una componente dell'azione.
- Se la configurazione articolare compare nello stesso contesto dell'istruzione linguistica, usare $\boldsymbol{\theta}$ per i giunti in modo da evitare ambiguità con $q$.
- Mantenere la notazione coerente tra formule, testo e figure lungo tutto il capitolo.
