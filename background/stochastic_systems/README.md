# Sistemi stocastici e simulazione

Questa sezione raccoglie gli appunti su **simulazione a eventi discreti, teoria delle code e modelli di inventario**. Gli argomenti condividono con il Reinforcement Learning l'uso di processi aleatori, stato, dinamiche temporali e criteri di prestazione, ma costituiscono un percorso autonomo di modellazione e ricerca operativa.

La separazione evita di identificare la simulazione con il RL. Un simulatore descrive l'evoluzione di un sistema sotto regole assegnate; un agente RL utilizza invece l'interazione, reale o simulata, per apprendere una strategia decisionale.

## Simulazione a eventi discreti

La **simulazione** permette di studiare sistemi la cui evoluzione analitica è difficile da ricavare. Nei modelli a eventi discreti, lo stato cambia in corrispondenza di eventi quali arrivi, partenze, guasti o consegne. L'[introduzione alla simulazione](01_simulazione/README.md) distingue sistema e modello e descrive le fasi di uno studio simulativo.

## Teoria delle code

I **sistemi di accodamento** descrivono entità che richiedono un servizio a una o più risorse. Distribuzioni degli interarrivi, tempi di servizio, capacità e disciplina della coda determinano ritardi, congestione e utilizzo. Il capitolo sulla [teoria delle code](02_teoria_delle_code/README.md) introduce processi di arrivo, notazione di Kendall, legge di Little e modelli markoviani.

Il [sistema a singolo server](03_coda_singolo_server/README.md) sviluppa una formulazione più circoscritta e ricava le principali misure di prestazione, tra cui attesa media, lunghezza della coda e utilizzazione.

## Modelli di inventario

La gestione delle scorte combina decisioni di riordino, domanda incerta, lead time e costi. L'approfondimento sui [modelli di inventario](04_modelli_di_inventario/README.md) parte dall'Economic Order Quantity e introduce punto di riordino, domanda composta, policy $(s,S)$, backlog e funzione di costo.

