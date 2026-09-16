# Fondamenti matematici dei Diffusion Models

Nel capitolo introduttivo abbiamo visto operativamente cosa succede:

- durante il training prendiamo un dato reale $x_0$, lo perturbiamo con rumore e chiediamo al modello di predire qualcosa che permetta di invertire la perturbazione;
- durante la generazione partiamo invece da rumore casuale e utilizziamo ripetutamente il modello per arrivare a un campione appartenente alla distribuzione dei dati.

Ora vogliamo rispondere alla domanda fondamentale:

> Perché imparare a rimuovere rumore permette di imparare una distribuzione di probabilità?

Questa domanda è il cuore teorico dei diffusion model.

La risposta richiede collegare quattro concetti:

$$
\boxed{
\text{aggiunta di rumore}
\longrightarrow
\text{distribuzioni }p_t(x)
\longrightarrow
\text{score }\nabla_x \log p_t(x)
\longrightarrow
\text{generazione}
}
$$

Vedremo inoltre perché la classica predizione del rumore

$$
\epsilon_\theta(x_t,t)
$$

è strettamente collegata alla stima dello score.

---

# 1. I dati come distribuzione

Quando diciamo che un modello generativo deve "imparare le immagini", dobbiamo essere un po' più precisi.

Supponiamo che un'immagine sia rappresentata da un vettore

$$
x\in\mathbb{R}^d.
$$

Per un'immagine RGB di dimensione $H\times W$,

$$
d=3HW.
$$

Un dataset contiene campioni:

$$
x^{(1)},x^{(2)},\ldots,x^{(N)}.
$$

Idealmente immaginiamo che questi esempi siano stati estratti da una distribuzione sconosciuta:

$$
x_0\sim p_{\text{data}}(x).
$$

Questa distribuzione è ciò che vorremmo imparare.

Un generative model dovrebbe quindi permetterci di produrre nuovi campioni

$$
\hat x\sim p_\theta(x)
$$

tali che

$$
p_\theta(x)\approx p_{\text{data}}(x).
$$

Il modello non deve semplicemente memorizzare le immagini di training.

Deve imparare una distribuzione dalla quale sia possibile campionare.

---

# 2. Perché modellare direttamente $p_{\text{data}}(x)$ è difficile

In teoria potremmo tentare di imparare direttamente

$$
p_{\text{data}}(x).
$$

Ma lo spazio delle immagini è estremamente grande.

Un'immagine può essere vista come un punto in uno spazio ad altissima dimensionalità.

Per esempio, un'immagine

$$
512\times512\times3
$$

contiene

$$
786\,432
$$

componenti.

La maggior parte dei punti di questo enorme spazio non rappresenta però immagini naturali.

Un vettore casuale di pixel produce tipicamente rumore.

Le immagini naturali occupano invece una regione enormemente più strutturata dello spazio.

Possiamo visualizzare schematicamente:

```text
spazio di tutte le possibili immagini

+---------------------------------------+
|                                       |
|           punti casuali               |
|                                       |
|        .                              |
|                    .                  |
|              __________               |
|             /          \              |
|            / immagini   \             |
|           |  naturali    |            |
|            \            /             |
|             \__________/              |
|                                       |
|      .                         .      |
+---------------------------------------+
```

La distribuzione dei dati

$$
p_{\text{data}}(x)
$$

è quindi estremamente complessa.

Una delle idee centrali dei diffusion model è:

> invece di imparare direttamente questa distribuzione complicata, costruiamo una sequenza di distribuzioni progressivamente più semplici.

---

# 3. Dal dato al rumore

Definiamo una successione

$$
x_0,x_1,\ldots,x_T.
$$

Il punto iniziale è

$$
x_0\sim p_{\text{data}}.
$$

Progressivamente aggiungiamo rumore.

Otteniamo quindi distribuzioni

$$
p_0(x),p_1(x),\ldots,p_T(x).
$$

Dove

$$
p_0(x)=p_{\text{data}}(x).
$$

Per $t$ crescente, la struttura dei dati viene progressivamente distrutta.

Idealmente, alla fine:

$$
p_T(x)\approx\mathcal N(0,I).
$$

Abbiamo quindi trasformato una distribuzione estremamente complicata in una distribuzione semplicissima.

```text
p_data
  ↓
p_1
  ↓
p_2
  ↓
...
  ↓
p_T ≈ N(0,I)
```

Questo processo prende il nome di **forward diffusion process**.

Il punto fondamentale è che questo processo non deve essere imparato.

Lo scegliamo noi.

---

# 4. Il forward process come catena di Markov

Nel DDPM classico il processo forward viene definito come una catena di Markov:

$$
q(x_t|x_{t-1}).
$$

La proprietà di Markov significa che

$$
x_t
$$

dipende direttamente soltanto dallo stato precedente

$$
x_{t-1}.
$$

La transizione viene definita come

$$
q(x_t|x_{t-1})
=
\mathcal N
\left(
x_t;
\sqrt{1-\beta_t}\,x_{t-1},
\beta_t I
\right).
$$

Introduciamo

$$
\alpha_t=1-\beta_t.
$$

Possiamo allora scrivere

$$
q(x_t|x_{t-1})
=
\mathcal N
\left(
x_t;
\sqrt{\alpha_t}x_{t-1},
(1-\alpha_t)I
\right).
$$

Equivalentemente, possiamo campionare

$$
x_t
=
\sqrt{\alpha_t}x_{t-1}
+
\sqrt{1-\alpha_t}\epsilon,
$$

con

$$
\epsilon\sim\mathcal N(0,I).
$$

Questa formula contiene due operazioni.

La componente precedente viene attenuata:

$$
x_{t-1}\rightarrow\sqrt{\alpha_t}x_{t-1},
$$

e viene aggiunta una componente casuale:

$$
\sqrt{1-\alpha_t}\epsilon.
$$

---

# 5. Il significato di $\beta_t$

Il parametro

$$
\beta_t
$$

controlla quanto rumore viene introdotto al passo $t$.

La sequenza

$$
\beta_1,\beta_2,\ldots,\beta_T
$$

viene chiamata **noise schedule**.

Se

$$
\beta_t
$$

è piccolo, ogni singolo passo distrugge soltanto una piccola quantità di informazione.

Tipicamente:

$$
0<\beta_t\ll1.
$$

Dopo molti passi, però, l'effetto cumulativo diventa molto grande.

Questo distingue due concetti che è utile non confondere:

$$
\beta_t
$$

descrive il rumore aggiunto in un singolo passaggio,

mentre la quantità di rumore complessivamente presente in $x_t$ dipende da tutti i passaggi precedenti.

---

# 6. Dalla catena alla formula chiusa

Simulare

$$
x_0\rightarrow x_1\rightarrow\ldots\rightarrow x_t
$$

durante ogni iterazione di training sarebbe molto inefficiente.

Fortunatamente non è necessario.

Definiamo

$$
\bar\alpha_t
=
\prod_{s=1}^{t}\alpha_s.
$$

Allora si può dimostrare che

$$
q(x_t|x_0)
=
\mathcal N
\left(
x_t;
\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I
\right).
$$

Pertanto possiamo scrivere direttamente:

$$
\boxed{
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon
}
$$

con

$$
\epsilon\sim\mathcal N(0,I).
$$

Questa è una delle formule più importanti dell'intera teoria dei diffusion model.

Conviene saperla ricostruire e interpretare.

---

# 7. Interpretazione della formula

Consideriamo

$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon.
$$

Ci sono due componenti.

## Segnale

$$
\sqrt{\bar\alpha_t}x_0.
$$

## Rumore

$$
\sqrt{1-\bar\alpha_t}\epsilon.
$$

Per timestep piccoli,

$$
\bar\alpha_t\approx1,
$$

quindi

$$
x_t\approx x_0.
$$

Il campione contiene quasi completamente il dato originale.

Per timestep grandi,

$$
\bar\alpha_t\approx0,
$$

quindi

$$
x_t\approx\epsilon.
$$

Il campione è quasi completamente rumore gaussiano.

Schematicamente:

```text
t = 0

signal ████████████████████
noise

t medio

signal ███████████
noise  ░░░░░░░░░

t grande

signal ██
noise  ░░░░░░░░░░░░░░░░░░
```

Questa interpretazione tornerà continuamente quando parleremo di signal-to-noise ratio.

---

# 8. Non esiste una singola distribuzione rumorosa

Questo punto è importante.

Non abbiamo soltanto

$$
p_{\text{data}}
$$

e

$$
\mathcal N(0,I).
$$

Abbiamo una famiglia continua o discreta di distribuzioni:

$$
p_t(x).
$$

Per ogni livello di rumore esiste quindi una distribuzione diversa.

Possiamo pensarla geometricamente come:

```text
distribuzione dati

      x   x
   x xxxxx x
      xxx

        ↓ rumore

     x x x x
   x  x x   x
      x x

        ↓

  x   x    x   x
     x   x
 x      x       x

        ↓

Gaussian-like cloud
```

La struttura complessa dei dati viene progressivamente "smussata".

Questo fenomeno sarà fondamentale quando introdurremo lo **score**.

---

# 9. La domanda inversa

Il forward process è facile:

$$
p_{\text{data}}
\longrightarrow
\mathcal N(0,I).
$$

Per generare dati dobbiamo fare il contrario:

$$
\mathcal N(0,I)
\longrightarrow
p_{\text{data}}.
$$

Se conoscessimo esattamente le distribuzioni coinvolte, potremmo considerare le transizioni inverse

$$
q(x_{t-1}|x_t).
$$

Il problema è che queste dipendono implicitamente dalla distribuzione dei dati.

Ed è proprio qui che entra in gioco il modello neurale.

Definiamo una distribuzione parametrica

$$
p_\theta(x_{t-1}|x_t)
$$

che cerca di approssimare il processo inverso.

Nel DDPM:

$$
p_\theta(x_{t-1}|x_t)
=
\mathcal N
\left(
x_{t-1};
\mu_\theta(x_t,t),
\Sigma_t
\right).
$$

Il network deve quindi fornire l'informazione necessaria per determinare la direzione nella quale muoversi da $x_t$ verso campioni progressivamente più plausibili.

---

# 10. Arriva lo score

Introduciamo adesso uno degli oggetti matematici più importanti:

$$
\boxed{
s(x)=\nabla_x\log p(x)
}
$$

Questa quantità viene chiamata **score function**.

Per una distribuzione dipendente dal tempo:

$$
s_t(x)
=
\nabla_x\log p_t(x).
$$

Attenzione: qui "score" non significa punteggio o qualità.

È semplicemente il gradiente rispetto a $x$ del logaritmo della densità di probabilità.

---

# 11. Interpretazione geometrica dello score

Consideriamo una distribuzione bidimensionale.

```text
             alta densità

                 ●
              ↗ ↑ ↖
             →  ●  ←
              ↘ ↓ ↙

       bassa densità
```

In un punto $x$, il vettore

$$
\nabla_x\log p(x)
$$

indica localmente la direzione nella quale la densità cresce più rapidamente.

In termini intuitivi:

> lo score indica in quale direzione dovremmo muovere un punto per portarlo verso regioni più probabili della distribuzione.

Questo è esattamente il tipo di informazione che serve durante la generazione.

Se partiamo da un punto rumoroso, vogliamo sapere in quale direzione modificarlo affinché diventi più compatibile con la distribuzione dei dati.

---

# 12. Esempio: score di una Gaussiana

Consideriamo

$$
p(x)
=
\mathcal N(x;\mu,\sigma^2I).
$$

Ignorando le costanti:

$$
\log p(x)
=
-\frac{1}{2\sigma^2}
\|x-\mu\|^2+C.
$$

Derivando rispetto a $x$:

$$
\nabla_x\log p(x)
=
-\frac{x-\mu}{\sigma^2}.
$$

Quindi

$$
\boxed{
s(x)
=
-\frac{x-\mu}{\sigma^2}
}
$$

Il vettore punta verso

$$
\mu.
$$

Se siamo lontani dalla regione ad alta probabilità, lo score indica come tornare verso di essa.

```text
                  x
                 ↙

            ↘         ↙

                 μ

            ↗         ↖
```

Questo piccolo esempio contiene già l'intuizione fondamentale della generazione score-based.

---

# 13. Score delle distribuzioni rumorose

Nel diffusion model siamo interessati a

$$
p_t(x_t).
$$

Per ogni timestep vorremmo conoscere

$$
\nabla_{x_t}\log p_t(x_t).
$$

Il network può quindi essere visto come una funzione

$$
s_\theta(x_t,t)
$$

che cerca di approssimare

$$
\nabla_{x_t}\log p_t(x_t).
$$

In altre parole:

$$
\boxed{
s_\theta(x_t,t)
\approx
\nabla_{x_t}\log p_t(x_t)
}
$$

Il timestep è necessario perché il campo vettoriale cambia con il livello di rumore.

La direzione corretta quando l'immagine è quasi pulita non è la stessa direzione corretta quando il campione è quasi completamente rumore.

---

# 14. Ma nei DDPM non prediciamo $\epsilon$?

Sì.

Ed è qui che arriva uno dei collegamenti più importanti.

Nel training classico utilizziamo

$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon.
$$

Il network riceve

$$
(x_t,t)
$$

e cerca di predire

$$
\epsilon.
$$

Quindi:

$$
\epsilon_\theta(x_t,t)
\approx\epsilon.
$$

La loss è tipicamente

$$
\mathcal L
=
\mathbb E_{x_0,t,\epsilon}
\left[
\|\epsilon-
\epsilon_\theta(x_t,t)\|^2
\right].
$$

Perché questo dovrebbe avere qualcosa a che fare con lo score?

---

# 15. Il collegamento tra rumore e score

Condizionatamente a $x_0$, sappiamo che

$$
q(x_t|x_0)
=
\mathcal N
\left(
\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I
\right).
$$

Per una Gaussiana abbiamo appena visto che lo score è

$$
\nabla_x\log p(x)
=
-\frac{x-\mu}{\sigma^2}.
$$

In questo caso:

$$
\mu
=
\sqrt{\bar\alpha_t}x_0
$$

e

$$
\sigma_t^2
=
1-\bar\alpha_t.
$$

Quindi

$$
\nabla_{x_t}
\log q(x_t|x_0)
=
-
\frac{
x_t-\sqrt{\bar\alpha_t}x_0
}{
1-\bar\alpha_t
}.
$$

Ma dalla formula di forward diffusion:

$$
x_t-\sqrt{\bar\alpha_t}x_0
=
\sqrt{1-\bar\alpha_t}\epsilon.
$$

Sostituendo:

$$
\nabla_{x_t}
\log q(x_t|x_0)
=
-
\frac{
\sqrt{1-\bar\alpha_t}\epsilon
}{
1-\bar\alpha_t
}.
$$

Otteniamo:

$$
\boxed{
\nabla_{x_t}
\log q(x_t|x_0)
=
-
\frac{\epsilon}
{\sqrt{1-\bar\alpha_t}}
}
$$

Questa relazione è fondamentale.

Predire

$$
\epsilon
$$

equivale, a meno di un fattore noto dipendente dal timestep, a predire lo score.

In forma schematica:

$$
\boxed{
\text{noise prediction}
\quad\Longleftrightarrow\quad
\text{score estimation}
}
$$

---

# 16. Perché questa relazione è così importante

Il network sembra essere allenato con un compito molto semplice:

> predici il rumore che ho aggiunto.

Ma matematicamente sta imparando qualcosa di molto più profondo.

Sta imparando un campo vettoriale definito sullo spazio dei dati:

$$
(x,t)
\mapsto
s_\theta(x,t).
$$

Questo campo descrive localmente come muoversi verso regioni di maggiore probabilità.

Possiamo quindi reinterpretare il diffusion model non semplicemente come:

```text
noise remover
```

ma come:

```text
x_t
 ↓
neural network
 ↓
direzione locale
 ↓
regione più plausibile della distribuzione
```

Questa interpretazione sarà essenziale quando passeremo a:

- score-based diffusion;
- probability flow ODE;
- Continuous Normalizing Flows;
- Flow Matching;
- Rectified Flow.

---

# 17. Un'importante precisazione

Bisogna fare attenzione a una sottigliezza.

Abbiamo appena calcolato

$$
\nabla_{x_t}\log q(x_t|x_0),
$$

cioè lo score della distribuzione rumorosa **condizionata al dato originale**.

Durante la generazione però non conosciamo $x_0$.

Ci interessa invece:

$$
\nabla_{x_t}\log p_t(x_t).
$$

Il risultato fondamentale del denoising score matching è che, minimizzando opportunamente l'errore medio sui campioni rumorosi, possiamo imparare proprio lo score marginale necessario alla generazione.

Non è quindi necessario conoscere $x_0$ durante inference.

Durante il training $x_0$ viene utilizzato per costruire un target supervisionato conveniente.

Una volta allenato, il network utilizza soltanto

$$
x_t
$$

e

$$
t.
$$

Nel caso condizionato utilizzerà inoltre una condizione $c$:

$$
s_\theta(x_t,t,c).
$$

---

# 18. Il quadro concettuale completo

Possiamo adesso riscrivere tutto il diffusion model in maniera più precisa.

## Forward

Partiamo da

$$
x_0\sim p_{\text{data}}.
$$

Aggiungiamo progressivamente rumore:

$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon.
$$

Questo genera una famiglia di distribuzioni:

$$
p_0,p_1,\ldots,p_T.
$$

Con:

$$
p_0=p_{\text{data}},
$$

e

$$
p_T\approx\mathcal N(0,I).
$$

## Training

Il network impara informazioni equivalenti allo score:

$$
s_\theta(x_t,t)
\approx
\nabla_{x_t}\log p_t(x_t).
$$

Nel DDPM classico questo viene fatto indirettamente predicendo:

$$
\epsilon_\theta(x_t,t).
$$

## Generazione

Partiamo da:

$$
x_T\sim\mathcal N(0,I).
$$

Utilizziamo il campo imparato per muoverci attraverso la sequenza:

$$
p_T
\rightarrow
p_{T-1}
\rightarrow
\cdots
\rightarrow
p_0.
$$

Alla fine:

$$
x_0\sim p_{\text{data}}
$$

approssimativamente.

---

# 19. Una nuova immagine mentale

La rappresentazione:

```text
rumore → denoise → denoise → immagine
```

è utile, ma ora possiamo sostituirla con una più precisa.

```text
                 learned vector field

                       ↑
                       |
Gaussian noise  →  intermediate distributions  →  data
     p_T                p_t                     p_0
```

Il modello ha imparato, per diversi valori di $t$, come orientarsi all'interno dello spazio.

Questa idea di **campo vettoriale dipendente dal tempo** è il ponte fondamentale verso Flow Matching.

---

# 20. Cosa bisogna sapere a questo punto

Prima di continuare, dovresti essere in grado di spiegare senza formule davanti:

1. perché un generative model cerca di modellare una distribuzione e non una singola immagine;
2. perché aggiungere rumore rende progressivamente più semplice la distribuzione;
3. cosa rappresentano $\beta_t$, $\alpha_t$ e $\bar\alpha_t$;
4. perché possiamo ottenere direttamente $x_t$ da $x_0$;
5. cosa rappresenta

$$
\nabla_x\log p(x);
$$

6. perché lo score può essere interpretato come una direzione locale;
7. perché la predizione di $\epsilon$ è collegata alla predizione dello score;
8. perché il timestep deve essere fornito al network;
9. cosa significa dire che il diffusion model impara un campo vettoriale;
10. perché questa visione prepara naturalmente il passaggio a Flow Matching.

Le formule che conviene invece conoscere molto bene sono:

$$
\alpha_t=1-\beta_t,
$$

$$
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s,
$$

$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon,
$$

$$
s_t(x)
=
\nabla_x\log p_t(x),
$$

e

$$
\nabla_{x_t}\log q(x_t|x_0)
=
-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}.
$$

Il prossimo passo consiste nel rendere rigoroso il **reverse diffusion process** e derivare il DDPM: vedremo da dove nasce

$$
p_\theta(x_{t-1}|x_t),
$$

perché viene modellato come una Gaussiana, come entra la ELBO e perché alla fine l'obiettivo di training può essere ridotto alla semplice loss MSE sulla predizione del rumore.