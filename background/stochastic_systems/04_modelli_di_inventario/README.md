# Modelli di inventario

## EOQ

La formula di Wilson (o EOQ, **Economic Order Quantity**) è un modello matematico di gestione delle scorte che serve a calcolare la quantità ottimale di merce da ordinare per ridurre al minimo i costi di stoccaggio e di ordinazione.

La formula è la seguente:

$$
EOQ = \sqrt{\frac{2DS}{H}}
$$

dove $D$ è la domanda annua, $S$ è il costo fisso per ogni ordine e $H$ è il costo di mantenimento (nel magazzino) per unità all'anno.

Il modello EOQ si basa su alcune ipotesi semplificative:

- La domanda è **costante** (e prevedibile)
- I **tempi di consegna** (lead time) sono **costanti**
- **Non sono ammessi backorder** (ordini arretrati)
- **Costi di ordinazione e di mantenimento sono costanti**
- **Consegna immediata**

In altre parole, il modello EOQ permette di calcolare la quantità ottimale da ordinare in un contesto ideale, ma non tiene conto delle incertezze che possono verificarsi nella realtà, come variazioni della domanda o ritardi nelle consegne.

Per esempio, se un'azienda ha una domanda annua di 10,000 unità, un costo fisso per ordine di 50 euro e un costo di mantenimento di 2 euro per unità all'anno, la quantità ottimale da ordinare sarebbe:

$$
EOQ = \sqrt{\frac{2 \cdot 10000 \cdot 50}{2}} = \sqrt{500000} \approx 707 \text{ unità}
$$

### Punto di Riordino

Il modello di Wilson specifica solo la quantità ottimale da ordinare, ma non indica quando effettuare l'ordine. Per questo motivo, si introduce il concetto di **punto di riordino** (Reorder Point, ROP), che rappresenta il livello di inventario al quale è necessario effettuare un nuovo ordine per evitare rotture di stock.

Seguendo la logica del modello EOQ, il punto di riordino può essere calcolato come:

$$
ROP = d \cdot L
$$

dove $d$ è la domanda media giornaliera e $L$ è il lead time (tempo di consegna) in giorni. In questo modo, quando l'inventario scende al di sotto del ROP, l'azienda effettua un nuovo ordine per garantire che le scorte siano sufficienti fino all'arrivo della nuova fornitura.

Per esempio, se la domanda media giornaliera è di 100 unità e il lead time è di 5 giorni, il punto di riordino sarebbe:

$$
ROP = 100 \cdot 5 = 500 \text{ unità}
$$

## Incertezza Stocastica

Nei contesti applicativi reali, la domanda e i tempi di fornitura non sono mai perfettamente stabili. La simulazione avanzata di un magazzino deve modellare due fonti primarie di stocasticità:

- Domanda Composta (Compound Demand)
- Rischio Pipeline (Lead Time Variabile)

### Compound Demand

Piuttosto che un deflusso lineare costante, l'arrivo dei clienti segue un andamento casuale. I *tempi di interarrivo* (la distanza temporale tra due vendite successive) sono modellati come variabili casuali Indipendenti e Identicamente Distribuite (IID) estratte da una distribuzione Esponenziale.

Contemporaneamente, la quantità richiesta ad ogni transazione varia secondo una distribuzione discreta indipendente. Ad esempio:

- 1 unità con probabilità 1/6
- 2 unità con probabilità 1/3
- 3 unità con probabilità 1/3
- 4 unità con probabilità 1/6

Questa struttura matematica cattura in modo formale l'aggregazione di clienti che si presentano a intervalli imprevedibili, acquistando lotti di grandezza disomogenea.

### Lead Time Variabile

Il momento in cui l'ordine viene emesso e quello in cui i prodotti diventano utilizzabili a scaffale differiscono a causa del **lead time di fornitura**.

Il **Lead Time** rappresenta il ritardo (delivery lag) tra il momento in cui l'ordine viene emesso al fornitore e il momento in cui la merce è fisicamente disponibile in magazzino. Nel sistema, tale ritardo è descritto da una distribuzione Uniforme continua compresa tra 0.5 e 1 mese.

Trattando questo ritardo come una variabile Uniforme (ad esempio, fluttuante tra 0.5 e 1 mese), il magazzino è esposto al **Pipeline Risk**. Questo descrive la vulnerabilità dell'azienda durante il periodo di transito, in cui picchi di domanda repentini non possono essere compensati, rendendo matematicamente indispensabile il ricorso a scorte di sicurezza (safety stock).



## Policy di Ripristino (s, S)

**Stationary $(s, S)$ Replenishment Policy** è una logica di controllo periodico. All'inizio di ogni mese, l'azienda ispeziona il livello dell'inventario $I$ e decide la quantità $Z$ da ordinare dal fornitore. La quantità da ordinare $Z$ è determinata dalla seguente funzione definita a tratti:


$$
Z = \begin{cases} S - I & \text{se } I < s \\ 0 & \text{se } I \ge s \end{cases}
$$

In letteratura, questa politica è spesso denominata policy "Min-Max". Il parametro $s$ agisce come punto di riordino (Reorder Point, ROP): la soglia critica che innesca l'azione di approvvigionamento.

Il parametro $S$ funge da livello obiettivo (Order-Up-To Level). Dal punto di vista della teoria matematica dell'inventario (dimostrata originariamente da Herbert Scarf nel 1960), la policy $(s, S)$ è dimostrabilmente ottimale per sistemi soggetti a costi fissi di setup, poiché ammortizza il costo dell'ordine ordinando grandi lotti fino a raggiungere il livello $S$.

### Gestione del Backlog e Inventario Netto

Se la domanda al tempo $t$, $D_t$, è minore o uguale all'inventario fisico, viene soddisfatta immediatamente ($I(t) \ge D_t$). Se invece la domanda eccede le disponibilità fisiche ($I(t) < D_t$), la porzione non soddisfatta viene posta in ***backlog*** (domanda arretrata) e verrà evasa prioritariamente non appena arriverà una consegna futura.

Di conseguenza, la variabile di stato dell'inventario $I(t)$ rappresenta l'**Inventario Netto**, il quale può assumere valori negativi; infatti $I(t) < 0$ indica esattamente l'ammontare del backlog. Quando avviene una consegna, i pezzi ricevuti coprono prima i backlog e, per il rimanente, accrescono la giacenza fisica positiva.

### Struttura dei Costi (Objective Function)

L'obiettivo manageriale è identificare la combinazione dei parametri $(s, S)$ che minimizza il costo totale medio mensile, espresso come somma di tre vettori di costo:

$$
C_{tot} = C + H + S
$$

#### Costi di Ordinazione

I **costi di ordinazione** $C$ (Ordering costs) si manifestano esclusivamente quando viene emesso un ordine, includendo una componente fissa e una variabile:


$$
C = K + i \cdot Z
$$

- **Costo di Setup** $K$ rappresenta l'onere amministrativo, logistico e di trasporto necessario per processare un ordine, indipendente dalla sua dimensione. È l'elemento che spinge l'azienda ad accorpare gli ordini.


- **Costo Incrementale** $i$ è il costo unitario di acquisto o produzione del singolo pezzo ordinato.



#### Costi di Mantenimento

Il mantenimento a scorta comporta oneri di varia natura: affitto del magazzino, assicurazioni, tasse, deperimento e, soprattutto, il costo opportunità del capitale (WACC) immobilizzato nei prodotti stoccati.

I **costi di mantenimento** $H$ vengono applicati solo all'inventario fisico presente in magazzino, definito tramite la parte positiva della funzione inventario: $I^+(t) = \max\{I(t), 0\}$.

Il costo di mantenimento medio si calcola come:


$$
H = h \cdot \hat{I}^+
$$


Dove $h = 1$ è il costo unitario di mantenimento e $\hat{I}^+$ rappresenta la *media temporale continua* (time-average) dell'inventario positivo lungo gli $n$ mesi di simulazione. Dal punto di vista geometrico e analitico, equivale a:


$$
\hat{I}^+ = \frac{\int_0^n I^+(t) dt}{n}
$$


L'integrale calcola esattamente l'area sottesa dalle "curve a dente di sega" dell'inventario fisico nel tempo.

#### Costi di Rottura di Stock

Quando l'azienda non riesce a soddisfare la domanda, incorre in penali. Nel caso del backlog, questi costi coprono l'amministrazione dell'arretrato e la perdita di "goodwill" (fiducia e soddisfazione) del cliente.

Questi costi si applicano sulla porzione negativa dell'inventario, ossia il volume degli ordini inevasi: $I^-(t) = \max\{-I(t), 0\}$.

Il costo di shortage medio è:


$$
S = \pi \cdot \hat{I^-}
$$


Dove $\pi$ indica la penale unitaria per ogni pezzo in arretrato per unità di tempo, e la media temporale del backlog è l'integrale sulle aree di deficit:


$$
\hat{I^-} = \frac{\int_0^n I^-(t) dt}{n}
$$


## Tecniche Classiche di Forecasting (Previsione della Domanda)

Per alimentare modelli come il sistema $(s, S)$ è imprescindibile prevedere accuratamente le serie storiche della domanda.

Tradizionalmente, gli approcci statistici classici hanno dominato questo campo:

- **Medie Mobili** (Moving Averages): Una tecnica lineare che attenua le fluttuazioni casuali, calcolando la media aritmetica degli $N$ periodi precedenti. Pur essendo stabili, reagiscono molto lentamente ai cambi strutturali di trend.
- **Smussamento Esponenziale** (Holt-Winters): Applica pesi decrescenti in modo esponenziale ai dati passati, conferendo maggior rilevanza agli eventi recenti. Modelli come l'Holt-Winters permettono di modellare sia il trend di crescita sia gli schemi di stagionalità ciclica.
- **Modelli ARIMA**: (AutoRegressive Integrated Moving Average) Esplorano la dipendenza statistica (autocorrelazione) tra i ritardi temporali passati della domanda. Pur essendo precisi su orizzonti stabili, risultano inefficaci con profili di vendita fortemente non-lineari.