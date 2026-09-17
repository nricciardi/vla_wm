# Imitation Learning e Behavioral Cloning

Un agente robotico deve trasformare ciò che percepisce in azioni che producano un comportamento desiderato. In linea di principio, questo comportamento può essere appreso specificando una funzione di ricompensa e ottimizzandola mediante Reinforcement Learning. In molti problemi reali, tuttavia, definire una ricompensa corretta è difficile: una specifica incompleta può premiare comportamenti indesiderati, mentre raccogliere interazioni sufficienti direttamente su un robot può essere lento, costoso o pericoloso.

L'**Imitation Learning (IL)** segue una strada differente. Invece di descrivere il comportamento attraverso una funzione di ricompensa, lo mostra mediante un insieme di **dimostrazioni** prodotte da un esperto. L'obiettivo consiste nell'apprendere una policy capace di riprodurre il comportamento osservato e, soprattutto, di eseguirlo autonomamente in nuovi episodi.

Il **Behavioral Cloning (BC)** è la forma più semplice e diffusa di Imitation Learning. Tratta ogni coppia osservazione-azione presente nelle dimostrazioni come un esempio supervisionato:

$$
\text{osservazione dell'esperto}
\longrightarrow
\text{azione dell'esperto}.
$$

Questa formulazione è alla base di gran parte del robot learning moderno. Molti Vision-Language-Action model, pur differendo per scala, architettura e rappresentazione delle azioni, vengono addestrati inizialmente proprio come policy di Behavioral Cloning su grandi collezioni di traiettorie robotiche.

## Il problema dell'Imitation Learning

Consideriamo un agente che interagisce con un ambiente a passi discreti. Al tempo $t$, l'ambiente si trova in uno stato $s_t \in \mathcal{S}$, l'agente riceve un'osservazione $o_t \in \mathcal{O}$ e sceglie un'azione $a_t \in \mathcal{A}$. La dinamica dell'ambiente determina quindi lo stato successivo:

$$
s_{t+1} \sim P(\,s_{t+1} \mid s_t, a_t\,).
$$

Lo **stato** contiene tutte le informazioni necessarie a descrivere l'ambiente, mentre l'**osservazione** contiene soltanto ciò che è accessibile all'agente. In simulazione i due possono coincidere; su un robot reale, invece, lo stato completo del mondo non è normalmente disponibile. Una camera può non mostrare un oggetto occluso, i sensori hanno rumore e una singola immagine non permette sempre di stimare velocità o intenzioni.

La policy descrive come vengono selezionate le azioni. Una policy stocastica osservazionale è una distribuzione condizionata:

$$
\pi(a_t \mid o_t).
$$

Se la singola osservazione non è sufficiente, la decisione può dipendere dalla storia disponibile fino al tempo $t$:

$$
h_t = (o_0,a_0,o_1,a_1,\ldots,o_t),
\qquad
\pi(a_t \mid h_t).
$$

Nella robotica language-conditioned, l'istruzione $l$ entra a sua volta nel contesto della policy:

$$
\pi(a_t \mid h_t,l).
$$

Questa distinzione è importante: scrivere $\pi(a_t\mid o_t)$ non implica che ogni problema sia realmente risolvibile a partire da un singolo frame. È soltanto una scelta di modellazione, spesso adottata per semplificare la notazione.

### Esperto, dimostrazione e traiettoria

Si assume di avere accesso a un **esperto**, descritto concettualmente da una policy $\pi_E$. L'esperto può essere:

- una persona che teleopera il robot;
- un controller progettato manualmente;
- un planner che dispone dello stato completo dell'ambiente;
- una policy precedentemente addestrata in simulazione o mediante Reinforcement Learning;
- un insieme eterogeneo di operatori, robot e procedure di raccolta.

Eseguendo la policy esperta si ottengono traiettorie della forma

$$
\tau^{(i)} =
(o_0^{(i)},a_0^{(i)},o_1^{(i)},a_1^{(i)},\ldots,o_{T_i}^{(i)}).
$$

Un dataset di dimostrazioni può essere scritto come

$$
\mathcal{D}_E = \{\tau^{(1)},\tau^{(2)},\ldots,\tau^{(N)}\}.
$$

Una **transizione** descrive un singolo passo dell'interazione, per esempio $(o_t,a_t,o_{t+1})$; una **traiettoria** è invece la sequenza ordinata di transizioni che compone un episodio. Durante il training del BC le coppie $(o_t,a_t)$ possono essere campionate individualmente, ma non diventano per questo dati privi di struttura temporale: sono state generate da un sistema dinamico e la loro distribuzione dipende dalla policy che ha raccolto le traiettorie.

Il dataset può inoltre contenere istruzioni linguistiche, stato propriocezionale, immagini provenienti da più camere, reward, indicatori di successo o altre annotazioni:

$$
\tau^{(i)} =
(l^{(i)},o_0^{(i)},a_0^{(i)},\ldots,o_{T_i}^{(i)},a_{T_i}^{(i)}).
$$

Il Behavioral Cloning utilizza necessariamente osservazioni e azioni; le altre informazioni diventano utili soltanto se entrano nel condizionamento della policy, nella loss oppure nella selezione dei dati.

### Imitation Learning non coincide con Behavioral Cloning

I due termini non sono sinonimi. **Imitation Learning** indica la famiglia generale di metodi che apprendono dalle dimostrazioni. Al suo interno rientrano, tra gli altri:

- **Behavioral Cloning**, che imita direttamente le azioni dell'esperto;
- **interactive imitation learning**, in cui l'esperto fornisce nuove etichette o correzioni sugli stati visitati dalla policy appresa;
- **Inverse Reinforcement Learning**, che cerca di inferire una funzione di ricompensa compatibile con il comportamento osservato;
- metodi adversarial o occupancy-matching, che cercano di rendere simili le distribuzioni di visita dell'esperto e dell'agente.

Il BC è quindi una possibile soluzione al problema dell'imitazione, non la sua definizione completa.

## Behavioral Cloning come apprendimento supervisionato

Sia $\pi_\theta$ una policy parametrizzata, per esempio da una rete neurale con parametri $\theta$. Il principio del Behavioral Cloning consiste nello scegliere i parametri che rendono probabili le azioni osservate nel dataset:

$$
\theta^*
=
\arg\max_\theta
\sum_{\tau\in\mathcal{D}_E}
\sum_{t=0}^{T_\tau}
\log \pi_\theta(a_t\mid o_t).
$$

In modo equivalente, si minimizza la negative log-likelihood:

$$
\mathcal{L}_{\mathrm{BC}}(\theta)
=
-\mathbb{E}_{(o,a)\sim\mathcal{D}_E}
\left[
\log \pi_\theta(a\mid o)
\right].
$$

Con istruzioni linguistiche e memoria, il condizionamento può essere esteso senza modificare il principio:

$$
\mathcal{L}_{\mathrm{BC}}(\theta)
=
-\mathbb{E}_{(h,l,a)\sim\mathcal{D}_E}
\left[
\log \pi_\theta(a\mid h,l)
\right].
$$

Questa forma probabilistica è più generale delle loss implementate concretamente. Cross-entropy e mean squared error emergono scegliendo differenti distribuzioni per $\pi_\theta$.

### Azioni discrete

Se l'azione appartiene a un insieme finito, la policy può produrre una distribuzione categorica. Per esempio:

$$
\mathcal{A}
=
\{\texttt{left},\texttt{right},\texttt{open},\texttt{close}\}.
$$

La rete produce un logit $z_k$ per ogni azione e il softmax lo converte in una probabilità:

$$
\pi_\theta(a=k\mid o)
=
\frac{\exp(z_k)}{\sum_j \exp(z_j)}.
$$

Minimizzare la negative log-likelihood equivale in questo caso a utilizzare la cross-entropy:

$$
\mathcal{L}_{\mathrm{CE}}
=
-\frac{1}{M}\sum_{i=1}^{M}
\log \pi_\theta(a_i^E\mid o_i),
$$

dove $M$ è il numero di esempi del batch e $a_i^E$ è l'azione dimostrata dall'esperto.

Anche azioni originariamente continue possono essere discretizzate e rappresentate come token. Questa scelta permette di riutilizzare l'obiettivo autoregressivo dei language model, ma introduce una risoluzione finita e richiede di definire intervalli e bin di quantizzazione.

### Azioni continue

Nel controllo robotico, un'azione contiene spesso quantità continue come target articolari, velocità, coppie o spostamenti dell'end-effector:

$$
a_t =
[\Delta x,\Delta y,\Delta z,
\Delta r_x,\Delta r_y,\Delta r_z,
g]_t,
$$

dove $g$ rappresenta, per esempio, il comando del gripper.

Una formulazione comune usa una policy deterministica $\hat a=f_\theta(o)$ e minimizza il mean squared error:

$$
\mathcal{L}_{\mathrm{MSE}}(\theta)
=
\frac{1}{M}
\sum_{i=1}^{M}
\left\|f_\theta(o_i)-a_i^E\right\|_2^2.
$$

La MSE non è una scelta neutra. Corrisponde alla maximum likelihood di una distribuzione gaussiana isotropa con varianza fissata:

$$
\pi_\theta(a\mid o)
=
\mathcal{N}\!\left(a;\mu_\theta(o),\sigma^2 I\right).
$$

Minimizzare la sua negative log-likelihood, ignorando i termini costanti, produce infatti

$$
-\log \pi_\theta(a^E\mid o)
\propto
\left\|a^E-\mu_\theta(o)\right\|_2^2.
$$

Questa interpretazione chiarisce l'ipotesi implicita: per ogni osservazione, le azioni valide vengono descritte da una singola moda gaussiana. Quando l'ipotesi non è appropriata, una bassa MSE non garantisce azioni sensate.

### Ambiguità e multimodalità

Da uno stesso stato possono esistere più azioni corrette. Per afferrare un oggetto, per esempio, il robot potrebbe aggirare un ostacolo da sinistra oppure da destra. Se nel dataset compaiono entrambe le strategie, una regressione deterministica può predirne la media:

$$
a_{mathrm{pred}}
\approx
\frac{a_{\mathrm{left}}+a_{\mathrm{right}}}{2}.
$$

L'azione media può non corrispondere ad alcun comportamento valido e condurre direttamente verso l'ostacolo. Questo fenomeno non indica necessariamente una scarsa capacità della rete: può derivare dal fatto che la loss e la distribuzione scelta per la policy non rappresentano adeguatamente la multimodalità dei dati.

Possibili soluzioni includono:

- mixture density network, che rappresentano più componenti probabilistiche;
- discretizzazione e predizione autoregressiva delle azioni;
- latent-variable model, che introducono una variabile latente per separare strategie differenti;
- diffusion policy e flow-based policy, che apprendono distribuzioni continue complesse;
- un contesto temporale o linguistico più informativo, che elimina parte dell'ambiguità apparente.

L'ultima possibilità è concettualmente diversa dalle precedenti. Se due azioni appaiono incompatibili soltanto perché il modello non osserva l'intenzione dell'operatore o una parte rilevante della scena, il problema non è esclusivamente la distribuzione di output: manca una variabile di condizionamento.

## Training offline ed esecuzione closed loop

Il training standard del Behavioral Cloning è **offline**: il dataset è già stato raccolto e l'ottimizzazione dei parametri non modifica gli stati che contiene. La loss viene calcolata su osservazioni visitate dall'esperto.

Durante un rollout, invece, la policy è inserita nel loop dinamico dell'ambiente:

$$
o_t
\xrightarrow{\pi_\theta}
a_t
\xrightarrow{P}
s_{t+1}
\longrightarrow
o_{t+1}.
$$

Ogni azione influenza l'osservazione successiva. Un errore apparentemente piccolo può quindi spostare il robot in uno stato che non compare nelle dimostrazioni. Da quello stato la policy è meno affidabile, può commettere un errore più grande e allontanarsi ulteriormente dalla distribuzione di training.

### La distribuzione degli input dipende dalla policy

Per evitare di assumere che lo stato completo sia osservabile, indichiamo con $x_t$ l'input effettivamente fornito alla policy: può essere uno stato, una singola osservazione oppure una storia di osservazioni e azioni. Sia $d_t^\pi(x)$ la distribuzione di questi input al tempo $t$ quando viene eseguita la policy $\pi$. Il BC minimizza un errore sotto la distribuzione generata dall'esperto:

$$
\mathbb{E}_{x\sim d_t^{\pi_E}}
\left[
\ell(\pi_\theta(x),\pi_E(x))
\right].
$$

Al momento dell'esecuzione, ciò che interessa è invece l'errore sotto la distribuzione indotta dalla policy appresa:

$$
\mathbb{E}_{x\sim d_t^{\pi_\theta}}
\left[
\ell(\pi_\theta(x),\pi_E(x))
\right].
$$

In generale,

$$
d_t^{\pi_E}(x) \neq d_t^{\pi_\theta}(x).
$$

Questa differenza è chiamata **distribution shift** o, più specificamente nel contesto del BC, **covariate shift indotto dalla policy**. Le azioni corrette non cambiano necessariamente significato; cambia la distribuzione delle osservazioni sulle quali la policy deve operare.

### Accumulo degli errori

Supponiamo, in modo semplificato, che la policy abbia probabilità $\epsilon$ di scegliere un'azione errata sugli stati dell'esperto. Se gli errori portano verso regioni mai viste durante il training, non è possibile assumere che il tasso di errore rimanga $\epsilon$ per tutto l'episodio.

Nell'analisi classica che motiva DAgger, sotto specifiche ipotesi, il costo aggiuntivo del Behavioral Cloning può crescere quadraticamente con l'orizzonte $T$:

$$
J(\pi_\theta)-J(\pi_E)=O(T^2\epsilon).
$$

Il termine non deve essere interpretato come una legge universale valida per ogni robot. Esprime il meccanismo fondamentale: un errore precoce può alterare molti passi futuri e il costo del singolo errore può crescere con il tempo rimanente nell'episodio.

Un esempio semplice è un robot che deve mantenere l'end-effector sopra una traiettoria stretta. Nel dataset tutte le correzioni partono da posizioni vicine alla traiettoria ideale. Durante l'esecuzione, una piccola deviazione laterale produce un'osservazione fuori distribuzione; se il dataset non contiene esempi di recupero, la policy non ha mai imparato quale azione applicare in quella regione.

### Perché la validation loss non basta

Una validation loss calcolata su frame estratti dalle dimostrazioni misura la capacità di imitare l'esperto sulla sua distribuzione. Non misura direttamente:

- se gli errori producono stati recuperabili;
- se una sequenza di azioni completa il task;
- se il robot rimane stabile per un intero episodio;
- se la policy generalizza a nuovi oggetti, ambienti o istruzioni;
- se le azioni predette sono sicure quando il sistema esce dalla traiettoria nominale.

Due policy con la stessa MSE possono avere performance closed-loop molto diverse. Una può commettere errori innocui in tratti facilmente correggibili; l'altra può sbagliare raramente, ma proprio nei punti decisivi del task.

## Strategie per rendere il BC più robusto

I limiti del Behavioral Cloning non implicano che il metodo sia inutilizzabile. Indicano quali proprietà devono essere curate nel dataset, nel modello e nella procedura di valutazione.

### Aumentare copertura e diversità delle dimostrazioni

Un dataset utile non dovrebbe contenere soltanto esecuzioni nominali quasi identiche. Variazioni nelle pose iniziali, negli oggetti, nello sfondo, nell'illuminazione e nelle strategie dell'operatore aumentano la regione dello spazio osservazionale coperta dalle dimostrazioni.

Sono particolarmente preziose le **recovery demonstration**, nelle quali l'esperto mostra come tornare verso una configurazione valida dopo una perturbazione. La semplice aggiunta di frame visivamente differenti, tuttavia, non garantisce copertura dal punto di vista del controllo: le variazioni devono rappresentare stati che la policy può realmente incontrare.

### Perturbazioni e data augmentation

Rumore sensoriale, augmentation visive e piccole perturbazioni dello stato possono ridurre la sensibilità a variazioni irrilevanti. Devono però preservare la relazione semantica tra osservazione e azione. Se un'immagine viene riflessa orizzontalmente, per esempio, può essere necessario riflettere anche le componenti dell'azione e modificare eventuali istruzioni che distinguono destra e sinistra.

L'augmentation non sostituisce automaticamente nuovi dati di interazione: trasformare un'immagine non insegna necessariamente come recuperare da uno stato fisico mai dimostrato.

### Imitation Learning interattivo

Metodi come **DAgger** eseguono iterativamente la policy corrente, chiedono all'esperto quale azione sarebbe corretta sugli stati visitati dall'agente e aggregano le nuove coppie al dataset:

```text
dataset iniziale dell'esperto
        ↓
training della policy
        ↓
rollout della policy appresa
        ↓
label dell'esperto sugli stati visitati
        ↓
aggregazione dei dati e nuovo training
```

In questo modo il training si avvicina alla distribuzione di stati che la policy incontrerà durante l'esecuzione. Il vantaggio richiede però accesso ripetuto all'esperto e introduce problemi pratici di sicurezza: una policy ancora imperfetta deve essere eseguita, almeno parzialmente, nell'ambiente.

### Policy espressive e contesto temporale

Distribuzioni multimodali, memoria, action chunking e architetture con capacità sufficiente riducono errori che non dipendono dalla sola copertura del dataset. È utile tenere separati i problemi:

- una policy unimodale può mediare tra comportamenti validi differenti;
- una policy senza memoria può non distinguere stati percettivamente identici ma dinamicamente diversi;
- una policy addestrata soltanto su stati nominali può fallire fuori distribuzione anche se l'architettura è molto espressiva.

Cambiare modello risolve soltanto il limite che quel modello introduce; non crea informazione assente dalle dimostrazioni.

## Aspetti pratici nella robotica

Prima ancora dell'architettura, una pipeline di Behavioral Cloning dipende da decisioni che stabiliscono il significato effettivo del problema supervisionato.

### Sincronizzazione

Osservazioni e azioni devono essere temporalmente allineate. A causa di latenza di rete, acquisizione delle camere e dinamica degli attuatori, il comando registrato allo stesso timestamp di un'immagine potrebbe essere stato deciso usando un'immagine precedente. Un disallineamento sistematico induce la policy ad apprendere reazioni in ritardo.

### Rappresentazione e normalizzazione delle azioni

È necessario dichiarare se le azioni rappresentano:

- posizione, velocità, coppia o incremento di posa;
- coordinate assolute o relative;
- joint space o task space;
- orientazioni espresse mediante angoli di Eulero, quaternioni o altre parametrizzazioni;
- target per un controller oppure comandi inviati direttamente agli attuatori.

Componenti con scale molto differenti possono dominare una loss non pesata. Le statistiche di normalizzazione devono essere calcolate soltanto sul training set e riutilizzate senza modificarle durante validation e deployment.

### Frequenza di controllo

La stessa sequenza numerica di azioni assume un significato differente a frequenze diverse. Una policy addestrata a $10\,\mathrm{Hz}$ non può essere eseguita automaticamente a $50\,\mathrm{Hz}$ senza ridefinire durata dei comandi, action chunk e interazione con il controller sottostante.

### Suddivisione del dataset

Frame consecutivi della stessa traiettoria sono fortemente correlati. Una suddivisione casuale per frame può collocare immagini quasi identiche sia nel training sia nella validation, producendo una stima eccessivamente ottimistica.

È preferibile suddividere almeno per traiettoria. Per misurare forme specifiche di generalizzazione, si possono inoltre mantenere separati task, oggetti, configurazioni iniziali, ambienti o operatori.

### Dati non uniformi

Le traiettorie possono avere lunghezze diverse e alcune fasi, come l'avvicinamento a un oggetto, possono occupare molti più timestep di eventi brevi ma decisivi, come la chiusura del gripper. Campionare uniformemente tutti i frame assegna implicitamente più peso alle fasi più lunghe.

Mask per sequenze con padding, pesi per componente o fase, bilanciamento tra task e campionamento gerarchico tra dataset sono quindi parte della definizione dell'obiettivo, non meri dettagli implementativi.

## Valutare una policy imitativa

Una valutazione completa separa almeno tre livelli.

### Metriche offline

Misurano la predizione su dimostrazioni non usate durante il training:

- negative log-likelihood;
- cross-entropy o accuracy per azioni discrete;
- MSE, MAE o errore angolare per azioni continue;
- errore per componente dell'azione e per orizzonte futuro.

Sono utili per diagnosticare l'ottimizzazione e confrontare modelli sullo stesso dataset, ma non sostituiscono i rollout.

### Metriche closed-loop

Misurano il comportamento prodotto dall'interazione tra policy e ambiente:

- success rate del task;
- return, se è disponibile una ricompensa informativa;
- tempo o numero di passi necessari;
- collisioni, violazioni di sicurezza e interventi umani;
- frequenza e qualità dei recuperi dopo una perturbazione.

Il success rate dovrebbe essere accompagnato dal numero di episodi e da intervalli di confidenza, specialmente quando l'esecuzione su robot reale limita la dimensione del campione.

### Generalizzazione e robustezza

Occorre dichiarare che cosa cambia rispetto al training: nuove pose iniziali, nuovi oggetti, nuovi sfondi, nuove formulazioni linguistiche oppure una loro combinazione. Riunire tutte queste condizioni sotto l'etichetta generica *unseen* rende difficile capire quale capacità sia stata effettivamente misurata.
