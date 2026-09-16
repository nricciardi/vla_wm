Il punto chiave è separare molto bene due cose:

* in training, il modello impara a rimuovere rumore;
* in inference/generazione, parte dal rumore e applica ripetutamente ciò che ha imparato, fino a ottenere un’immagine.

La cosa controintuitiva è che durante il training normalmente non gli si chiede di generare un’immagine completa da zero. Gli si dà invece un problema molto più semplice: “dato qualcosa che ho sporcato con del rumore, dimmi quale rumore ho aggiunto”.

### 1. L'idea generale

Supponiamo di avere questa immagine originale:

`x₀ = 🐱 immagine pulita`

Definiamo un processo artificiale che aggiunge progressivamente rumore:

```text
x₀       x₁       x₂       x₃       ...       x_T
🐱  →   🐱░  →   ░🐱░  →   ░░░░  →  ...  →   rumore
```

Dopo abbastanza passi, dell'immagine originale praticamente non rimane nulla.

Questo processo si chiama spesso forward diffusion.

Importante: non è qualcosa che la rete neurale deve imparare. Siamo noi a definire matematicamente come aggiungere il rumore.

Il modello impara invece il processo opposto:

```text
rumore → un po' meno rumore → ancora meno rumore → ... → 🐱
```

cioè il reverse diffusion process.

---

## Training

Prendiamo un'immagine reale dal dataset, per esempio un gatto.

Chiamiamola:

$$
x_0
$$

Poi scegliamo casualmente un timestep, per esempio:

$$
t=600
$$

Ora generiamo del rumore gaussiano casuale:

$$
\epsilon \sim N(0,I)
$$

e creiamo una versione rumorosa dell'immagine:

$$
x_t
$$

Per esempio:

```text
immagine originale                  immagine al passo 600

     🐱                  →              ▒░▓▒░
     x₀                                  x₆₀₀
```

Ma c'è un dettaglio fondamentale.

Noi sappiamo esattamente quale rumore abbiamo aggiunto.

Quindi possiamo dare alla rete:

```text
INPUT
    immagine rumorosa x_t
    timestep t
    eventualmente testo ("a photo of a cat")

                    ↓
              neural network
             (es. U-Net / DiT)

                    ↓

OUTPUT
    rumore previsto ε̂
```

e confrontarlo con il vero rumore $\epsilon$ che avevamo aggiunto.

La loss classica è grossomodo:

$$
L = ||\epsilon-\hat{\epsilon}||^2
$$

Quindi il task del modello è:

> “Guarda questa immagine rumorosa e dimmi qual è il rumore presente.”

Ripetendo questo su milioni/miliardi di immagini e su timestep diversi, il modello diventa molto bravo a riconoscere quale parte di un'immagine rumorosa sembra essere “segnale” e quale sembra essere “rumore”.

Ed è qui che, implicitamente, impara la distribuzione delle immagini.

Per sapere che cosa togliere da qualcosa come:

```text
▓░▒░▓▒░
```

deve aver imparato cose come:

“questa struttura è plausibile come occhio”
“questa è plausibile come pelliccia”
“questa forma potrebbe essere una faccia”
“questo pattern non sembra appartenere a un'immagine naturale”

Quindi non sta semplicemente imparando un filtro anti-rumore. Sta imparando moltissima struttura statistica del mondo visuale.

---

# Il trucco importante del training

Uno potrebbe pensare:

> “Per allenarlo al timestep 600 bisogna aggiungere rumore 600 volte?”

No.

Grazie alla matematica del processo di diffusione possiamo saltare direttamente da $x_0$ a $x_t$.

La formula semplificata è:

$$
x_t =
\sqrt{\bar{\alpha}_t} x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon
$$

Quindi durante un singolo esempio di training facciamo essenzialmente:

```text
prendo x₀
   ↓
scelgo t casuale
   ↓
genero ε casuale
   ↓
calcolo direttamente x_t
   ↓
rete(x_t, t) → ε̂
   ↓
confronto ε̂ con ε
   ↓
backpropagation
```

Non dobbiamo simulare tutta la catena.

Questo rende il training estremamente più efficiente.

---

# Inference

Qui avviene qualcosa di molto diverso.

Non abbiamo più un'immagine originale $x_0$.

Vogliamo crearne una nuova.

Quindi iniziamo direttamente da:

$$
x_T \sim N(0,I)
$$

cioè puro rumore casuale:

```text
▓░▒█░▒▓░█▒
```

Ora chiediamo al modello:

> “Dato che siamo al timestep T, quale rumore pensi ci sia qui dentro?”

Il modello produce:

$$
\hat{\epsilon}
$$

Usando quella previsione, il sampler calcola un'immagine leggermente meno rumorosa:

$$
x_T \rightarrow x_{T-1}
$$

Poi ripetiamo.

```text
x_T
puro rumore
    ↓
modello
    ↓
x_{T-1}
    ↓
modello
    ↓
x_{T-2}
    ↓
modello
    ↓
...
    ↓
x_1
    ↓
x_0
immagine
```

Quindi, a differenza del training, durante inference la procedura è iterativa.

---

## Un esempio concreto

Prompt:

> "a red sports car on a mountain road"

All'inizio:

```text
step 50

▓▒░█▓░▒█▒▓▒░
```

Il modello guarda il rumore insieme al testo.

Predice qualcosa del tipo:

> “Per ottenere qualcosa compatibile con ‘red sports car on a mountain road’, questa parte sembra rumore e dovrebbe essere rimossa/modificata.”

Dopo alcuni step:

```text
step 40

▒▒░   ███
░    █████      ░░
```

Cominciano ad apparire strutture molto grossolane.

Poi:

```text
step 25

       montagne
     /\/\/\/\/\
        _____
      _/_____\_
-----/_______\-----
```

Poi dettagli sempre più fini:

```text
step 10

auto
ruote
strada
montagne
riflessi
texture
```

fino all'immagine finale.

Non significa però che il modello “vede un'auto nascosta dentro il rumore”.

L'auto non era lì.

È la sequenza delle predizioni del modello che porta quel campione casuale verso una regione dello spazio delle immagini che assomiglia a ciò che il modello ha imparato essere una:

> “red sports car on a mountain road”.

---

# Training vs inference

Questa è probabilmente la distinzione più importante:

| Training                                    | Inference                                            |
| ------------------------------------------- | ---------------------------------------------------- |
| parto da un'immagine vera                   | parto da rumore casuale                              |
| aggiungo rumore                             | rimuovo progressivamente rumore                      |
| scelgo normalmente un timestep casuale      | percorro una sequenza di timestep                    |
| conosco il vero rumore $\epsilon$         | non conosco il “vero” rumore                         |
| il modello predice $\hat\epsilon$         | il modello predice $\hat\epsilon$                  |
| confronto $\hat\epsilon$ con $\epsilon$ | uso $\hat\epsilon$ per calcolare il prossimo stato |
| faccio backpropagation                      | niente backpropagation                               |
| aggiorno i pesi                             | pesi congelati                                       |

La stessa rete viene quindi usata in maniera molto diversa.

Durante training:

```text
x₀
 ↓
aggiungo artificialmente rumore
 ↓
x_t ───────→ MODELLO ───────→ ε̂
                               │
vero rumore ε ─────────────────┘
                 loss
                  ↓
            aggiorno pesi
```

Durante inference:

```text
x_T = rumore
     ↓
   MODELLO
     ↓
x_{T-1}
     ↓
   MODELLO
     ↓
x_{T-2}
     ↓
   MODELLO
     ↓
   ...
     ↓
x₀ = immagine
```

### E il prompt di testo?

Nei modelli text-to-image come Stable Diffusion, FLUX e modelli simili, il testo viene trasformato da un text encoder in una rappresentazione numerica.

Semplificando:

```text
"a small dog on the beach"
             ↓
        text encoder
             ↓
     vettori / embeddings
             ↓
              ┌───────────────┐
x_t ─────────→│ diffusion     │→ rumore previsto
t ───────────→│ model         │
text ────────→│               │
              └───────────────┘
```

Quindi il modello non deve semplicemente chiedersi:

> “Che cosa sembra un'immagine naturale?”

ma:

> “Che cosa sembra un'immagine naturale compatibile con questo testo?”

È questo che indirizza il processo di denoising verso un cane sulla spiaggia invece che, per esempio, un'automobile.

### Una cosa che spesso crea confusione: Stable Diffusion non fa tutto questo direttamente sui pixel

Stable Diffusion è un latent diffusion model.

Quindi in realtà il diagramma è più simile a:

```text
immagine
   ↓
VAE encoder
   ↓
latent z₀
   ↓
aggiungo rumore
   ↓
z_t
   ↓
diffusion model
```

Il processo di diffusione avviene in uno spazio compresso, detto latent space.

Durante inference:

```text
rumore latent
     ↓
denoising
     ↓
latent pulito
     ↓
VAE decoder
     ↓
immagine RGB
```

Per esempio, invece di lavorare direttamente su milioni di valori RGB, il modello lavora su una rappresentazione molto più compatta dell'immagine. Questo rende training e generazione molto meno costosi.

Se vuoi fissarti una sola immagine mentale, usa questa:

```text
                 TRAINING

   immagine vera
       ↓
  la sporco artificialmente
       ↓
   immagine rumorosa
       ↓
  «dimmi il rumore»
       ↓
     modello
       ↓
 confronto con il rumore
 che so di aver aggiunto
       ↓
    aggiorno modello


                INFERENCE

     rumore casuale
          ↓
       modello
          ↓
    meno rumore
          ↓
       modello
          ↓
    meno rumore
          ↓
        ...
          ↓
      immagine
```

La frase più precisa con cui riassumerei un diffusion model è quindi:

> Durante il training impara il campo di direzione che permette di trasformare distribuzioni rumorose in distribuzioni di dati; durante l'inference segue iterativamente quel campo partendo da un campione di rumore.

L'ultimo passaggio — capire perché “predire il rumore” equivale matematicamente a imparare la distribuzione delle immagini, e quindi perché dal puro rumore possa effettivamente emergere un'immagine — è il pezzo più interessante e meno intuitivo.
