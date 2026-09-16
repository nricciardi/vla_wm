# Diffusion Models: teoria, evoluzione, limiti e implementazione

## 1. Introduzione: che cosa sono i diffusion model

I diffusion model sono modelli generativi che imparano a trasformare una distribuzione semplice, tipicamente rumore gaussiano, nella distribuzione complessa dei dati.

L'idea fondamentale è dividere il problema generativo in molti problemi relativamente semplici.

Si consideri un'immagine reale

$$
x_0 \sim q_{\text{data}}(x).
$$

Definiamo due processi:

1. **forward diffusion process**: aggiunge progressivamente rumore a $x_0$;
2. **reverse diffusion process**: parte dal rumore e cerca di ricostruire progressivamente un campione appartenente alla distribuzione dei dati.

In forma schematica:

$$
x_0 \rightarrow x_1 \rightarrow x_2 \rightarrow \dots \rightarrow x_T
$$

per il processo forward e

$$
x_T \rightarrow x_{T-1} \rightarrow \dots \rightarrow x_1 \rightarrow x_0
$$

per il processo generativo.

Se $T$ è sufficientemente grande, $x_T$ diventa approssimativamente rumore gaussiano:

$$
x_T \approx \mathcal N(0,I).
$$

Il punto importante è che il processo forward viene scelto da noi ed è quindi noto. Il problema di machine learning consiste nell'imparare il processo inverso.

---

# 2. Deep Unsupervised Learning using Nonequilibrium Thermodynamics

**Sohl-Dickstein et al., ICML 2015**

Questo lavoro rappresenta uno dei fondamenti dei diffusion probabilistic models moderni. Gli autori propongono di modellare distribuzioni complesse attraverso un processo ispirato alla termodinamica fuori dall'equilibrio: la struttura presente nei dati viene progressivamente distrutta mediante diffusione, dopodiché viene appreso il processo inverso che ricostruisce la struttura.

## 2.1 Motivazione

Storicamente nei modelli probabilistici esiste un compromesso tra:

- flessibilità;
- trattabilità matematica.

Una distribuzione semplice come una Gaussiana è facile da normalizzare, campionare e valutare, ma non descrive immagini naturali.

Una distribuzione arbitrariamente complessa

$$
p(x)=\frac{\phi(x)}{Z}
$$

può essere molto espressiva, ma la costante di normalizzazione

$$
Z=\int \phi(x)dx
$$

può diventare impossibile da calcolare.

L'idea di Sohl-Dickstein è evitare di rappresentare direttamente una trasformazione estremamente complessa. La trasformazione viene scomposta in moltissimi piccoli passaggi.

---

## 2.2 Forward diffusion

Consideriamo una catena di Markov:

$$
q(x_{1:T}|x_0)
=
\prod_{t=1}^{T}q(x_t|x_{t-1}).
$$

Ogni transizione aggiunge una piccola quantità di rumore.

Nel caso gaussiano:

$$
q(x_t|x_{t-1})
=
\mathcal N
\left(
x_t;
\sqrt{1-\beta_t}x_{t-1},
\beta_tI
\right).
$$

Il parametro

$$
\beta_t \in (0,1)
$$

controlla quanto rumore viene aggiunto al timestep $t$.

Per valori piccoli di $\beta_t$:

$$
x_t \approx x_{t-1} + \text{piccolo rumore}.
$$

Iterando il processo molte volte, l'informazione relativa al campione iniziale viene progressivamente distrutta.

---

## 2.3 Reverse diffusion

Il processo inverso ideale sarebbe

$$
q(x_{t-1}|x_t).
$$

Questa distribuzione, però, dipende dalla distribuzione sconosciuta dei dati.

Si introduce quindi un modello parametrico:

$$
p_\theta(x_{t-1}|x_t).
$$

Nel caso gaussiano:

$$
p_\theta(x_{t-1}|x_t)
=
\mathcal N
\left(
x_{t-1};
\mu_\theta(x_t,t),
\Sigma_\theta(x_t,t)
\right).
$$

Una rete neurale deve quindi predire i parametri della trasformazione inversa.

---

## 2.4 Modello generativo

La distribuzione completa è

$$
p_\theta(x_{0:T})
=
p(x_T)
\prod_{t=1}^{T}
p_\theta(x_{t-1}|x_t),
$$

dove normalmente

$$
p(x_T)=\mathcal N(0,I).
$$

Per generare un'immagine:

1. campioniamo $x_T\sim\mathcal N(0,I)$;
2. stimiamo $x_{T-1}$;
3. stimiamo $x_{T-2}$;
4. continuiamo fino a ottenere $x_0$.

---

## 2.5 Training come problema variazionale

Il modello viene addestrato massimizzando indirettamente la likelihood dei dati.

Si utilizza una variational lower bound:

$$
-\log p_\theta(x_0)
\leq
\mathcal L_{\text{VLB}}.
$$

La loss può essere scomposta in termini associati alle singole transizioni della catena.

Questo è un elemento fondamentale: invece di imparare una trasformazione arbitrariamente complessa

$$
\text{rumore}\rightarrow\text{immagine},
$$

si imparano molti piccoli problemi di denoising.

---

## 2.6 Limiti

Il lavoro del 2015 introduce l'idea fondamentale, ma presenta diversi problemi pratici.

### Sampling molto lento

La generazione richiede potenzialmente centinaia o migliaia di passaggi.

### Formulazione complessa

L'obiettivo di training è matematicamente elegante, ma meno semplice da implementare rispetto ai diffusion model moderni.

### Qualità delle immagini

Nel 2015 i diffusion model non erano ancora competitivi con i migliori modelli generativi per immagini.

### Parametrizzazione del reverse process

Non era ancora chiaro quale fosse il modo migliore di parametrizzare la rete neurale.

---

# 3. Denoising Diffusion Probabilistic Models — DDPM

**Ho, Jain e Abbeel, 2020**

DDPM rende la formulazione dei diffusion model molto più semplice e pratica, mostrando inoltre una connessione importante con denoising score matching. Il paper ottiene risultati di alta qualità nella generazione di immagini e costituisce la base diretta di moltissime architetture successive.

---

# 3.1 Notazione

Poniamo

$$
\alpha_t=1-\beta_t.
$$

Definiamo inoltre

$$
\bar\alpha_t
=
\prod_{s=1}^{t}\alpha_s.
$$

Il processo forward è

$$
q(x_t|x_{t-1})
=
\mathcal N
\left(
\sqrt{\alpha_t}x_{t-1},
(1-\alpha_t)I
\right).
$$

Una proprietà estremamente importante è che possiamo calcolare direttamente $x_t$ a partire da $x_0$, senza simulare tutti i passaggi precedenti:

$$
q(x_t|x_0)
=
\mathcal N
\left(
\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I
\right).
$$

Di conseguenza:

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

Questa è probabilmente l'equazione più importante da ricordare dei DDPM.

---

# 3.2 Interpretazione

Quando $t$ è piccolo:

$$
\bar\alpha_t\approx1
$$

e quindi

$$
x_t\approx x_0.
$$

Quando $t$ è grande:

$$
\bar\alpha_t\approx0
$$

e quindi

$$
x_t\approx\epsilon.
$$

Il timestep controlla quindi il rapporto segnale/rumore.

---

# 3.3 Perché predire il rumore?

Una possibile rete potrebbe predire direttamente la media

$$
\mu_\theta(x_t,t).
$$

DDPM introduce però una parametrizzazione particolarmente efficace: far predire alla rete il rumore $\epsilon$ che è stato aggiunto.

La rete diventa:

$$
\epsilon_\theta(x_t,t).
$$

Durante il training conosciamo esattamente il rumore utilizzato per costruire $x_t$.

Possiamo quindi minimizzare:

$$
\boxed{
L_{\text{simple}}
=
\mathbb E_{x_0,t,\epsilon}
\left[
\|
\epsilon-
\epsilon_\theta(x_t,t)
\|^2
\right]
}
$$

Questa loss è estremamente semplice.

La rete riceve:

$$
(x_t,t)
$$

e deve rispondere:

> "Quale rumore è stato aggiunto a questa immagine?"

---

# 3.4 Algoritmo di training

Per ogni minibatch:

1. scegli un'immagine $x_0$;
2. scegli casualmente un timestep $t$;
3. genera rumore $\epsilon$;
4. costruisci $x_t$;
5. fai predire alla rete $\epsilon_\theta(x_t,t)$;
6. minimizza l'MSE.

---

# 3.5 Implementazione PyTorch del forward process

```python
import torch

T = 1000

betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

def extract(values, t, x_shape):
    out = values.to(t.device)[t]
    return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))

def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)

    sqrt_alpha_bar = extract(
        torch.sqrt(alpha_bar), t, x0.shape
    )

    sqrt_one_minus_alpha_bar = extract(
        torch.sqrt(1 - alpha_bar), t, x0.shape
    )

    xt = (
        sqrt_alpha_bar * x0
        + sqrt_one_minus_alpha_bar * noise
    )

    return xt, noise
```

Il punto concettuale fondamentale è che non stiamo applicando il rumore iterativamente.

Grazie alla proprietà gaussiana possiamo andare direttamente:

$$
x_0\rightarrow x_t.
$$

Questo rende il training molto efficiente.

---

# 3.6 Training loop minimale

```python
import torch.nn.functional as F

def training_step(model, x0):
    batch_size = x0.shape[0]

    t = torch.randint(
        0,
        T,
        (batch_size,),
        device=x0.device
    )

    noise = torch.randn_like(x0)

    xt, noise = q_sample(
        x0,
        t,
        noise
    )

    predicted_noise = model(xt, t)

    loss = F.mse_loss(
        predicted_noise,
        noise
    )

    return loss
```

Il modello utilizzato originariamente è tipicamente una **U-Net**.

La rete deve inoltre ricevere l'informazione sul timestep tramite una time embedding, spesso basata su sinusoidal embeddings.

---

# 3.7 Dal rumore predetto a $x_0$

Dall'equazione

$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon
$$

possiamo ricavare:

$$
\hat x_0
=
\frac{
x_t
-
\sqrt{1-\bar\alpha_t}
\epsilon_\theta(x_t,t)
}{
\sqrt{\bar\alpha_t}
}.
$$

La rete quindi, predicendo il rumore, implicitamente permette di stimare l'immagine pulita.

---

# 3.8 Reverse process

La distribuzione viene modellata come:

$$
p_\theta(x_{t-1}|x_t)
=
\mathcal N
(
\mu_\theta(x_t,t),
\sigma_t^2I
).
$$

Usando la parametrizzazione tramite rumore:

$$
\mu_\theta(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}
{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t)
\right).
$$

Dopodiché:

$$
x_{t-1}
=
\mu_\theta(x_t,t)
+
\sigma_tz,
$$

con

$$
z\sim\mathcal N(0,I).
$$

---

# 3.9 Sampling DDPM

Pseudo-implementazione:

```python
@torch.no_grad()
def sample_ddpm(model, shape, device):
    x = torch.randn(shape, device=device)

    for t_value in reversed(range(T)):

        t = torch.full(
            (shape[0],),
            t_value,
            device=device,
            dtype=torch.long
        )

        eps = model(x, t)

        alpha_t = alphas[t_value]
        alpha_bar_t = alpha_bar[t_value]
        beta_t = betas[t_value]

        mean = (
            1 / torch.sqrt(alpha_t)
        ) * (
            x -
            beta_t
            / torch.sqrt(1 - alpha_bar_t)
            * eps
        )

        if t_value > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * noise
        else:
            x = mean

    return x
```

---

# 3.10 Limiti dei DDPM

Il DDPM risolve gran parte della complessità concettuale del lavoro del 2015, ma lascia importanti problemi aperti.

## Sampling lento

Per generare una singola immagine possono essere richieste circa $T=1000$ valutazioni della U-Net.

Questa diventa una delle principali debolezze dei diffusion model.

## Noise schedule non ottimale

Il paper utilizza principalmente uno schedule lineare per $\beta_t$.

Non necessariamente tutti i timestep contribuiscono in modo altrettanto utile.

## Reverse variance

La varianza del processo inverso viene sostanzialmente fissata anziché imparata in modo ottimale.

## Likelihood

La qualità percettiva è molto buona, ma la likelihood non è ancora necessariamente competitiva con altri modelli.

Questi aspetti saranno affrontati da lavori successivi.

---

# 4. DDIM — Denoising Diffusion Implicit Models

**Song, Meng, Ermon, 2020**

Storicamente DDIM precede iDDPM. Lo introduco ora perché affronta direttamente il principale problema pratico dei DDPM: la lentezza del sampling.

DDIM mantiene lo stesso obiettivo di training del DDPM, ma costruisce una famiglia di processi forward **non-Markoviani** con le stesse distribuzioni marginali $q(x_t|x_0)$. Questo permette di utilizzare traiettorie inverse molto più corte. Gli autori riportano accelerazioni dell'ordine di 10–50× rispetto al sampling DDPM in alcuni esperimenti.

---

# 4.1 Il punto fondamentale

Un modello DDPM addestrato conosce approssimativamente la funzione

$$
\epsilon_\theta(x_t,t).
$$

DDIM osserva che non siamo obbligati a utilizzare esattamente la catena Markoviana usata nel DDPM per generare immagini.

Possiamo costruire un processo generativo alternativo che utilizza la stessa rete.

Quindi:

$$
\boxed{\text{stesso training, sampler differente}}
$$

Questo concetto sarà enormemente importante nella pratica.

---

# 4.2 Stima di $x_0$

Come prima:

$$
\hat x_0
=
\frac{
x_t-\sqrt{1-\bar\alpha_t}
\epsilon_\theta(x_t,t)
}{
\sqrt{\bar\alpha_t}
}.
$$

DDIM costruisce $x_{t-1}$ combinando:

- stima dell'immagine pulita;
- direzione del rumore;
- eventuale componente casuale.

Una forma comune è:

$$
x_{t-1}
=
\sqrt{\bar\alpha_{t-1}}\hat x_0
+
\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}
\epsilon_\theta(x_t,t)
+
\sigma_t z.
$$

---

# 4.3 Parametro $\eta$

La varianza può essere controllata da un parametro $\eta$.

### $\eta=1$

Il processo diventa più simile al sampling stocastico DDPM.

### $\eta=0$

Il processo diventa deterministico.

Quindi:

$$
x_T
\rightarrow x_0
$$

diventa una funzione deterministica per un determinato rumore iniziale.

Questo rende possibili proprietà interessanti come interpolazioni più consistenti.

---

# 4.4 Perché DDIM è più veloce?

Supponiamo che il DDPM sia stato addestrato con:

$$
T=1000.
$$

Non siamo obbligati a effettuare tutti i passaggi:

$$
999,998,997,\dots,1.
$$

Possiamo selezionare ad esempio:

$$
999,979,959,\dots
$$

e utilizzare solo 50 timestep.

La rete rimane la stessa.

---

# 4.5 Implementazione concettuale DDIM

```python
@torch.no_grad()
def ddim_step(model, x_t, t, t_prev):
    eps = model(x_t, t)

    a_t = alpha_bar[t]
    a_prev = alpha_bar[t_prev]

    x0_pred = (
        x_t
        - torch.sqrt(1 - a_t) * eps
    ) / torch.sqrt(a_t)

    x_prev = (
        torch.sqrt(a_prev) * x0_pred
        + torch.sqrt(1 - a_prev) * eps
    )

    return x_prev
```

Questa versione corrisponde concettualmente al caso deterministico $\eta=0$.

---

# 4.6 Limiti di DDIM

DDIM migliora enormemente la velocità, ma non risolve tutto.

### Riduzione degli step = possibile perdita di qualità

Con pochissimi step, la traiettoria numerica diventa più approssimata.

### Il training rimane quello del DDPM

DDIM migliora soprattutto il sampler, non modifica radicalmente il modello appreso.

### Costo della U-Net

Ogni singolo step continua comunque a richiedere una valutazione completa della rete.

Quindi il problema computazionale viene attenuato, non eliminato.

---

# 5. Improved DDPM — iDDPM

**Nichol e Dhariwal, 2021**

iDDPM non è semplicemente "un DDPM più grande". Il paper studia diversi dettagli del processo di diffusione e mostra che alcune modifiche relativamente semplici migliorano likelihood, training e sampling. Tra gli elementi principali troviamo varianza appresa, obiettivo ibrido e uno schedule del rumore basato sul coseno.

---

# 5.1 Problema dello schedule lineare

Nel DDPM:

$$
\beta_t
$$

segue tipicamente un andamento lineare.

Nichol e Dhariwal osservano che questo può distruggere l'informazione troppo velocemente, soprattutto per immagini a risoluzione relativamente bassa.

Introducono un **cosine noise schedule**.

Invece di specificare direttamente $\beta_t$, definiscono l'andamento di:

$$
\bar\alpha_t.
$$

Concettualmente:

$$
\bar\alpha_t
\approx
\frac{
f(t)
}{
f(0)
}
$$

con una funzione coseno.

Una forma semplificata è:

$$
f(t)
=
\cos^2
\left(
\frac{t/T+s}{1+s}
\frac{\pi}{2}
\right).
$$

Il risultato è una degradazione dell'informazione più graduale.

---

# 5.2 Learned variance

DDPM predice essenzialmente la media del reverse process mentre la varianza è fissata.

iDDPM permette al modello di imparare anche la varianza.

Il paper parametrizza la varianza interpolando nel dominio logaritmico tra due valori appropriati:

$$
\Sigma_\theta
=
\exp
\left(
v\log\beta_t
+
(1-v)
\log\tilde\beta_t
\right).
$$



La rete produce quindi anche $v$.

---

# 5.3 Hybrid objective

Il problema è che:

$$
L_{\text{simple}}
=
\|\epsilon-\epsilon_\theta\|^2
$$

non dipende dalla varianza predetta.

Viene quindi introdotta:

$$
L_{\text{hybrid}}
=
L_{\text{simple}}
+
\lambda L_{\text{VLB}}.
$$

Nel paper viene utilizzato un peso piccolo per evitare che la componente variazionale domini il training.

---

# 5.4 Importance sampling dei timestep

Un'altra osservazione è che i diversi timestep possono avere loss con varianze molto differenti.

Campionare uniformemente

$$
t\sim U(1,T)
$$

non è necessariamente ottimale.

Il paper studia strategie di importance sampling per dedicare maggiore attenzione ai timestep che contribuiscono maggiormente alla variabilità della loss.

---

# 5.5 Impatto sul sampling

Uno degli aspetti più importanti è che imparare la varianza permette di effettuare sampling utilizzando molti meno forward pass mantenendo una qualità simile. Gli autori riportano che la learned variance consente riduzioni dell'ordine di grandezza nel numero di passaggi con perdita limitata di qualità.

---

# 5.6 DDIM vs iDDPM

È importante non confonderli.

DDIM modifica principalmente:

$$
\boxed{\text{processo di sampling}}
$$

iDDPM modifica principalmente:

$$
\boxed{\text{training, noise schedule e reverse distribution}}
$$

Le due idee sono in larga misura complementari.

---

# 6. Il problema successivo: diffusion direttamente nei pixel

DDPM, DDIM e iDDPM dimostrano che la diffusione funziona estremamente bene.

Rimane però un problema strutturale.

Supponiamo di avere un'immagine:

$$
x\in\mathbb R^{512\times512\times3}.
$$

La U-Net deve processare centinaia di migliaia di variabili per ogni timestep.

E deve farlo magari:

$$
20,\ 50,\ 100,\ 1000
$$

volte per immagine.

Per immagini ad alta risoluzione questo diventa estremamente costoso.

Ed è precisamente questo il problema affrontato dai **Latent Diffusion Models**.

---

# 7. High-Resolution Image Synthesis with Latent Diffusion Models

**Rombach et al., 2021/2022**

Latent Diffusion Models, o LDM, spostano la diffusione dallo spazio dei pixel a uno spazio latente compresso ottenuto tramite un autoencoder.

Il paper nasce esplicitamente dal problema dell'elevato costo dei diffusion model che operano direttamente nello spazio dei pixel e introduce anche cross-attention per supportare conditioning flessibile come testo e bounding box.

---

# 7.1 Idea centrale

Introduciamo un encoder:

$$
E(x)=z.
$$

e un decoder:

$$
D(z)\approx x.
$$

Invece di effettuare:

$$
x_0
\rightarrow
x_1
\rightarrow\dots
\rightarrow x_T
$$

effettuiamo:

$$
z_0
\rightarrow
z_1
\rightarrow\dots
\rightarrow z_T.
$$

Dopo il sampling:

$$
z_0\rightarrow D(z_0)=x.
$$

---

# 7.2 Perché è più efficiente?

Supponiamo che l'immagine originale sia:

$$
512\times512\times3.
$$

L'encoder potrebbe produrre qualcosa come:

$$
64\times64\times4.
$$

La quantità di dati processata dalla rete diffusion diminuisce drasticamente.

Il modello può quindi dedicare la capacità computazionale alla struttura semanticamente importante anziché ricostruire continuamente dettagli pixel-level ridondanti.

---

# 7.3 Pipeline completa

## Training dell'autoencoder

$$
x
\xrightarrow{E}
z
\xrightarrow{D}
\hat x.
$$

L'autoencoder impara una rappresentazione compressa ma percettivamente fedele.

## Training della diffusion

Il diffusion model viene applicato su $z$:

$$
z_t
=
\sqrt{\bar\alpha_t}z_0
+
\sqrt{1-\bar\alpha_t}\epsilon.
$$

La loss rimane:

$$
L
=
\|
\epsilon-
\epsilon_\theta(z_t,t,c)
\|^2.
$$

Dove $c$ rappresenta l'eventuale conditioning.

---

# 7.4 Conditioning testuale

Un text encoder trasforma il prompt:

$$
y
\rightarrow
\tau(y).
$$

Per esempio:

> "a photograph of a red car"

diventa una sequenza di embedding.

Il diffusion model utilizza questi embedding tramite **cross-attention**.

---

# 7.5 Cross-attention

Nell'attention classica:

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V.
$$

Nella cross-attention:

- le query provengono dalle feature dell'immagine;
- key e value provengono dal conditioning testuale.

$$
Q=W_Q\phi(z_t)
$$

$$
K=W_K\tau(y)
$$

$$
V=W_V\tau(y).
$$

Poi:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt d}
\right)V.
$$

Questo consente alle feature visive di "consultare" il testo.

---

# 7.6 Esempio concettuale

Prompt:

> "a red cat sitting on a chair"

L'embedding potrebbe contenere token relativi a:

- red;
- cat;
- sitting;
- chair.

Durante diversi layer della U-Net, regioni differenti dell'immagine possono prestare attenzione a token differenti.

---

# 7.7 Pipeline PyTorch semplificata

```python
with torch.no_grad():
    z0 = vae.encode(images)

noise = torch.randn_like(z0)

zt = (
    sqrt_alpha_bar_t * z0
    + sqrt_one_minus_alpha_bar_t * noise
)

text_embeddings = text_encoder(tokens)

pred_noise = unet(
    zt,
    timestep=t,
    context=text_embeddings
)

loss = F.mse_loss(
    pred_noise,
    noise
)
```

In generazione:

```python
z = torch.randn(
    batch_size,
    latent_channels,
    latent_h,
    latent_w
)

for t in scheduler.timesteps:
    eps = unet(
        z,
        timestep=t,
        context=text_embeddings
    )

    z = scheduler.step(
        eps,
        t,
        z
    )

image = vae.decode(z)
```

Questa è, a livello concettuale, la struttura dietro sistemi come Stable Diffusion.

---

# 7.8 Limiti dei Latent Diffusion Models

LDM risolve gran parte del problema computazionale, ma introduce nuovi compromessi.

### Perdita di informazioni

Il diffusion model non vede l'immagine originale.

Vede:

$$
z=E(x).
$$

Se l'autoencoder elimina un dettaglio, il diffusion model non può necessariamente recuperarlo.

### Dipendenza dal VAE

La qualità finale è limitata anche dalla qualità di encoder e decoder.

### Artefatti

Dettagli fini, testi e pattern ad alta frequenza possono essere degradati dalla rappresentazione latente.

### Controllabilità incompleta

Il prompt testuale descrive **cosa** vogliamo.

È meno adatto a specificare esattamente:

- dove deve stare un oggetto;
- la posa di una persona;
- la profondità;
- i contorni;
- la struttura geometrica.

Questo porta a un nuovo ramo di ricerca: il **controllo spaziale**, di cui ControlNet è uno degli esempi principali.

Prima, però, consideriamo un altro problema: la personalizzazione.

---

# 8. Personalizzazione dei diffusion model

Un grande text-to-image model conosce concetti generici:

> dog

> car

> person

> chair

ma non conosce necessariamente:

> il mio cane specifico.

Immaginiamo di avere 4 fotografie di un oggetto chiamato informalmente $S$.

Vogliamo poter scrivere:

> "a photo of S wearing sunglasses"

senza riaddestrare il modello da zero.

Due lavori fondamentali sono:

1. Textual Inversion;
2. DreamBooth.

---

# 9. An Image is Worth One Word — Textual Inversion

**Gal et al., 2022**

Textual Inversion propone di rappresentare un nuovo concetto attraverso uno o pochi nuovi embedding testuali, lasciando congelato il diffusion model. Gli autori mostrano che 3–5 immagini possono essere utilizzate per apprendere una nuova "parola" nello spazio degli embedding del modello text-to-image.

---

# 9.1 Idea

Supponiamo che il text encoder abbia una embedding matrix:

$$
W
\in
\mathbb R^{V\times d}.
$$

Ogni token corrisponde a un vettore:

$$
w_i\in\mathbb R^d.
$$

Textual Inversion introduce un nuovo token:

$$
S^*
$$

e impara solamente il suo embedding:

$$
v^*.
$$

Tutto il resto rimane congelato.

---

# 9.2 Ottimizzazione

Abbiamo immagini:

$$
x_1,\dots,x_N.
$$

Usiamo prompt come:

> "a photo of $S^*$"

> "a painting of $S^*$"

> "$S^*$ on a table"

Durante il training ottimizziamo:

$$
v^*
$$

in modo che il diffusion model possa ricostruire correttamente il concetto.

Formalmente:

$$
v^*
=
\arg\min_v
\mathbb E
\left[
\|
\epsilon-
\epsilon_\theta
(
z_t,t,c(v)
)
\|^2
\right].
$$

La cosa importante è:

$$
\theta
$$

rimane congelato.

---

# 9.3 Codice concettuale

```python
for param in unet.parameters():
    param.requires_grad = False

for param in text_encoder.parameters():
    param.requires_grad = False

new_embedding.requires_grad = True

optimizer = torch.optim.Adam(
    [new_embedding],
    lr=5e-4
)

for images in dataset:

    z0 = vae.encode(images)

    noise = torch.randn_like(z0)
    t = sample_timesteps(z0.shape[0])

    zt = add_noise(z0, noise, t)

    text_embeddings = encode_prompt_with_new_token(
        prompt,
        new_embedding
    )

    predicted_noise = unet(
        zt,
        t,
        text_embeddings
    )

    loss = F.mse_loss(
        predicted_noise,
        noise
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

# 9.4 Vantaggi

### Numero di parametri piccolissimo

Possiamo ottimizzare poche migliaia di valori invece di centinaia di milioni.

### File personalizzato piccolo

Per memorizzare il concetto basta salvare l'embedding.

### Modello originale intatto

Non modifichiamo i pesi generativi.

---

# 9.5 Limiti

Il punto debole è anche la sua forza.

Stiamo cercando di comprimere un concetto visivo complesso in:

$$
v^*\in\mathbb R^d.
$$

Un singolo embedding può non contenere abbastanza informazione per preservare:

- identità precisa;
- geometria;
- texture;
- dettagli locali.

Textual Inversion tende quindi ad avere un compromesso:

$$
\text{editabilità}
\leftrightarrow
\text{fedeltà al soggetto}.
$$

Per concetti stilistici può funzionare molto bene, mentre per identità molto precise può risultare meno fedele.

---

# 10. DreamBooth

**Ruiz et al., 2022**

DreamBooth affronta lo stesso problema della personalizzazione ma sceglie una strategia molto più potente: invece di imparare soltanto un embedding, effettua fine-tuning del text-to-image diffusion model affinché associ un identificatore raro a uno specifico soggetto. Il paper utilizza poche immagini del soggetto e introduce una class-specific prior preservation loss per ridurre overfitting e language drift.

---

# 10.1 Textual Inversion vs DreamBooth

Textual Inversion:

$$
\boxed{\text{ottimizza il token}}
$$

DreamBooth:

$$
\boxed{\text{ottimizza il modello}}
$$

Questa differenza è fondamentale.

---

# 10.2 Identificatore del soggetto

Le immagini vengono associate a prompt come:

> "a [V] dog"

dove:

- `[V]` = identificatore raro;
- `dog` = classe del soggetto.

Il modello deve imparare:

$$
[V]+\text{dog}
\rightarrow
\text{quel cane specifico}.
$$

Il paper utilizza proprio una formulazione identifier + class noun.

---

# 10.3 Instance reconstruction loss

La loss principale rimane quella diffusion:

$$
L_{\text{instance}}
=
\mathbb E
\left[
\|
\epsilon-
\epsilon_\theta
(
z_t,t,c_{\text{instance}}
)
\|^2
\right].
$$

Con:

$$
c_{\text{instance}}
=
\text{"a [V] dog"}.
$$

---

# 10.4 Problema: language drift

Supponiamo di addestrare il modello solamente su fotografie del nostro cane.

Il modello potrebbe iniziare ad associare:

> dog

non più alla classe generale dei cani, ma quasi esclusivamente al cane personalizzato.

In termini intuitivi:

prima:

$$
\text{"dog"}
\rightarrow
\{\text{molti tipi di cani}\}
$$

dopo un fine-tuning eccessivo:

$$
\text{"dog"}
\rightarrow
\{\text{il nostro cane}\}.
$$

---

# 10.5 Prior preservation loss

DreamBooth introduce quindi una seconda loss.

Si generano immagini della classe generale usando il modello originale:

> "a dog"

ottenendo campioni

$$
x_{\text{prior}}.
$$

Si aggiunge:

$$
L_{\text{prior}}
=
\mathbb E
\left[
\|
\epsilon-
\epsilon_\theta
(
z_t^{\text{prior}},
t,
c_{\text{class}}
)
\|^2
\right].
$$

La loss totale è:

$$
\boxed{
L
=
L_{\text{instance}}
+
\lambda L_{\text{prior}}
}
$$

Lo scopo è preservare la conoscenza della classe generale e contrastare overfitting e language drift.

---

# 10.6 Esempio concettuale

```python
instance_loss = diffusion_loss(
    instance_images,
    instance_prompt
)

prior_loss = diffusion_loss(
    class_images,
    class_prompt
)

loss = (
    instance_loss
    + lambda_prior * prior_loss
)
```

---

# 10.7 Vantaggio rispetto a Textual Inversion

DreamBooth può modificare milioni di parametri.

Ha quindi molta più capacità per memorizzare:

- forma;
- colore;
- texture;
- dettagli caratteristici;
- identità del soggetto.

Tendenzialmente questo permette maggiore fedeltà al soggetto.

---

# 10.8 Limiti di DreamBooth

### Costo

Il fine-tuning è molto più pesante rispetto a Textual Inversion.

### Memoria

Se si personalizzano molti soggetti separatamente, salvare un modello completo per soggetto è inefficiente.

Questo problema contribuirà alla diffusione successiva di tecniche parameter-efficient come LoRA.

### Overfitting

Con poche immagini è relativamente facile memorizzare eccessivamente il training set.

### Identity vs prompt alignment

Aumentare troppo la fedeltà al soggetto può rendere più difficile modificarlo tramite prompt.

Nuovamente compare un trade-off:

$$
\text{subject fidelity}
\leftrightarrow
\text{text editability}.
$$

---

# 11. ControlNet — Adding Conditional Control to Text-to-Image Diffusion Models

**Zhang, Rao e Agrawala, 2023**

ControlNet affronta un problema differente.

Textual Inversion e DreamBooth chiedono:

> "Come posso insegnare al modello un nuovo soggetto?"

ControlNet chiede:

> "Come posso controllare precisamente la struttura spaziale dell'immagine?"

ControlNet aggiunge conditioning spaziale a grandi diffusion model text-to-image preaddestrati, mantenendo congelato il backbone originale e utilizzando una copia addestrabile collegata tramite convoluzioni inizializzate a zero. Il lavoro mostra conditioning tramite edge map, depth, segmentation, human pose e altri segnali.

---

# 11.1 Problema

Con un normale prompt:

> "a woman standing with her left arm raised"

non abbiamo una garanzia precisa sulla posa.

Il linguaggio è semanticamente potente ma geometricamente ambiguo.

Supponiamo invece di fornire uno skeleton OpenPose.

Vogliamo:

$$
p(x|\text{text},\text{pose})
$$

invece di:

$$
p(x|\text{text}).
$$

---

# 11.2 Principio architetturale

ControlNet parte da un modello già addestrato.

Una parte del modello viene:

$$
\boxed{\text{locked}}
$$

cioè congelata.

Viene creata una copia trainabile delle feature.

L'input condizionale viene elaborato e iniettato nella rete originale.

---

# 11.3 Zero convolution

Uno degli elementi distintivi è la **zero convolution**.

Si tratta essenzialmente di una convoluzione i cui pesi e bias vengono inizializzati a zero.

All'inizio:

$$
Z(x)=0.
$$

Quindi ControlNet inizialmente non altera il comportamento del modello originale.

Durante il training:

$$
Z(x)\neq0
$$

e la rete impara progressivamente l'effetto del controllo.

Questo evita che una rete di controllo appena inizializzata distrugga le feature di un modello già molto ben addestrato.

---

# 11.4 Forma concettuale

Supponiamo che:

$$
F(x;\theta)
$$

sia un blocco preaddestrato.

ControlNet produce qualcosa simile a:

$$
y
=
F(x;\theta)
+
Z_2
\left(
F_c(x+Z_1(c);\theta_c)
\right).
$$

Dove:

- $c$ è il controllo;
- $F$ è congelato;
- $F_c$ è trainabile;
- $Z_1,Z_2$ sono zero convolutions.

---

# 11.5 Pseudocodice

```python
class ControlBlock(nn.Module):
    def __init__(self, frozen_block, trainable_block):
        super().__init__()

        self.frozen = frozen_block
        self.trainable = trainable_block

        for p in self.frozen.parameters():
            p.requires_grad = False

        self.cond_zero = ZeroConv()
        self.out_zero = ZeroConv()

    def forward(self, x, condition):

        frozen_features = self.frozen(x)

        controlled_input = (
            x
            + self.cond_zero(condition)
        )

        control_features = self.trainable(
            controlled_input
        )

        return (
            frozen_features
            + self.out_zero(control_features)
        )
```

---

# 11.6 Tipi di controllo

ControlNet può usare:

### Edge

$$
c=\text{Canny}(x).
$$

### Depth

$$
c=\text{DepthEstimator}(x).
$$

### Pose

$$
c=\text{OpenPose}(x).
$$

### Segmentation

$$
c=\text{SemanticMap}(x).
$$

In tutti i casi il conditioning fornisce informazioni difficili da esprimere soltanto tramite linguaggio.

---

# 11.7 Limiti

### Richiede modelli specifici per controllo

Un ControlNet addestrato sulle pose non è automaticamente un ControlNet per depth map.

### Costo computazionale

Viene aggiunto un ramo importante alla U-Net.

L'inferenza è quindi più costosa rispetto al backbone da solo.

### Dipendenza dal preprocessor

Se la depth map o la pose sono sbagliate, il modello viene guidato verso una struttura sbagliata.

### Il conditioning non garantisce controllo perfetto

Il modello deve comunque conciliare:

- prompt;
- prior del diffusion model;
- segnale ControlNet.

---

# 12. Un problema architetturale: perché continuare a usare U-Net?

Fin qui quasi tutti i diffusion model per immagini utilizzano varianti della U-Net.

La U-Net ha molti vantaggi:

- convoluzioni efficienti;
- multi-resolution processing;
- skip connection;
- ottimo inductive bias per immagini.

Ma intorno al 2020–2022 i Transformer mostrano una caratteristica importante:

$$
\boxed{\text{scalabilità molto prevedibile con compute e dimensione}}
$$

Da qui nasce la domanda:

> È realmente necessaria una U-Net per un diffusion model?

---

# 13. Scalable Diffusion Models with Transformers — DiT

**Peebles e Xie, 2022/ICCV 2023**

DiT sostituisce la U-Net con un Transformer che opera su patch di rappresentazioni latenti. Gli autori studiano esplicitamente la scalabilità rispetto ai GFLOPs e osservano che modelli con maggiore capacità computazionale ottengono sistematicamente FID migliori; il modello DiT-XL/2 raggiunge nel paper un FID di 2.27 su ImageNet 256×256 con classifier-free guidance.

---

# 13.1 Punto di partenza: Latent Diffusion

DiT non torna necessariamente alla diffusione nei pixel.

Supponiamo di avere:

$$
z_t
\in
\mathbb R^{H\times W\times C}.
$$

Il latent viene diviso in patch, analogamente a Vision Transformer.

---

# 13.2 Patchify

Con patch di dimensione $p\times p$:

$$
N
=
\frac{H}{p}
\frac{W}{p}
$$

token.

Ogni patch viene linearizzata e proiettata:

$$
x_i
=
Wp_i+b.
$$

Otteniamo quindi:

$$
X
\in
\mathbb R^{N\times d}.
$$

A questo punto il problema di denoising può essere affrontato come sequence modeling.

---

# 13.3 Transformer block

Ogni blocco esegue operazioni del tipo:

$$
X'
=
X+\operatorname{Attention}(\operatorname{LN}(X))
$$

$$
X''
=
X'
+
\operatorname{MLP}(\operatorname{LN}(X')).
$$

Il modello deve però conoscere:

- timestep;
- classe o altro conditioning.

---

# 13.4 Conditioning tramite adaptive LayerNorm

Una delle soluzioni più efficaci studiate in DiT è l'adaptive LayerNorm.

La LayerNorm:

$$
\operatorname{LN}(x)
$$

viene modulata tramite parametri dipendenti dal conditioning:

$$
\operatorname{adaLN}(x,c)
=
\gamma(c)
\operatorname{LN}(x)
+
\beta(c).
$$

Dove $c$ contiene informazioni come:

$$
c=e_t+e_y
$$

con:

- $e_t$: timestep embedding;
- $e_y$: class embedding.

---

# 13.5 adaLN-Zero

DiT utilizza una variante chiamata **adaLN-Zero**.

L'idea ricorda parzialmente la filosofia delle zero convolution di ControlNet: inizializzare alcuni parametri di modulazione in modo che inizialmente il blocco abbia un comportamento vicino all'identità.

Questo tende a stabilizzare l'ottimizzazione.

---

# 13.6 Implementazione concettuale

```python
class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            dim,
            elementwise_affine=False
        )

        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(
            dim,
            elementwise_affine=False
        )

        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

        self.condition = nn.Linear(
            dim,
            6 * dim
        )

    def forward(self, x, c):

        shift1, scale1, gate1, \
        shift2, scale2, gate2 = \
            self.condition(c).chunk(6, dim=-1)

        h = self.norm1(x)
        h = h * (1 + scale1[:, None]) \
            + shift1[:, None]

        attn, _ = self.attn(h, h, h)

        x = x + gate1[:, None] * attn

        h = self.norm2(x)
        h = h * (1 + scale2[:, None]) \
            + shift2[:, None]

        x = (
            x
            + gate2[:, None] * self.mlp(h)
        )

        return x
```

È un'implementazione didattica, non la riproduzione completa del paper.

---

# 13.7 Scaling

Uno dei messaggi principali del lavoro è che aumentando:

- depth;
- width;
- numero di token;

aumentano i GFLOPs del modello e, empiricamente, la qualità migliora in modo piuttosto regolare.

Questo è importante perché porta i diffusion model verso una filosofia simile ai Large Language Models:

$$
\text{architettura semplice}
+
\text{molto compute}
+
\text{scaling}.
$$

---

# 13.8 Limiti di DiT

### Self-attention costosa

Il costo dell'attention standard è approssimativamente:

$$
O(N^2).
$$

Se aumentiamo il numero dei token, il costo cresce rapidamente.

### Richiede molto compute

Il fatto che un modello "scali bene" non significa che sia economico.

Al contrario, le prestazioni migliori richiedono modelli enormi.

### Inductive bias minore

La U-Net incorpora naturalmente strutture locali e multi-scale tipiche delle immagini.

Il Transformer deve impararne una parte dai dati.

### Il paper originale è class-conditional

Il lavoro di DiT non è originariamente il sistema text-to-image general purpose che a volte viene associato ai moderni diffusion transformer.

Il contributo fondamentale è soprattutto architetturale:

$$
\boxed{
\text{U-Net}
\rightarrow
\text{Transformer}
}
$$

nel contesto della latent diffusion.

---

# 14. La linea evolutiva completa

È utile ricostruire tutto attraverso i problemi affrontati.

## Sohl-Dickstein et al., 2015

Problema:

> Come modellare distribuzioni molto complesse mantenendo un processo generativo trattabile?

Soluzione:

$$
\text{distruzione progressiva}
+
\text{reverse process appreso}.
$$

Limite:

- formulazione complessa;
- qualità ancora limitata;
- molti step.

---

## DDPM, 2020

Problema:

> Come rendere questa idea pratica e ottenere immagini di alta qualità?

Soluzione:

$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon
$$

e prediction del rumore:

$$
\epsilon_\theta(x_t,t).
$$

Limite:

- sampling molto lento;
- variance e schedule migliorabili.

---

## DDIM, 2020

Problema:

> Dobbiamo davvero fare 1000 step di sampling?

Soluzione:

costruire un processo non-Markoviano compatibile con lo stesso training objective e saltare molti timestep.

Limite:

- la rete deve comunque essere valutata iterativamente;
- pochi step possono peggiorare la qualità.

---

## iDDPM, 2021

Problema:

> Possiamo migliorare likelihood, noise schedule, reverse variance e sampling mantenendo l'impostazione DDPM?

Soluzione:

- cosine schedule;
- learned variance;
- hybrid objective;
- importance sampling.

Limite:

- il diffusion model continua a essere computazionalmente costoso, soprattutto ad alta risoluzione.

---

## Latent Diffusion, 2021/2022

Problema:

> Perché fare denoising di milioni di pixel quando molta informazione è percettivamente ridondante?

Soluzione:

$$
x
\xrightarrow{encoder}
z
\xrightarrow{diffusion}
\hat z
\xrightarrow{decoder}
\hat x.
$$

Risultato:

molto meno compute.

Nuovo problema:

- il prompt offre controllo semantico ma non controllo geometrico preciso;
- il modello non conosce soggetti personali;
- il VAE introduce un information bottleneck.

---

## Textual Inversion, 2022

Problema:

> Come insegnare un nuovo concetto a un modello congelato?

Soluzione:

imparare:

$$
v^*
$$

cioè un nuovo embedding.

Vantaggio:

piccolissimo numero di parametri.

Limite:

capacità insufficiente per identità molto complesse.

---

## DreamBooth, 2022

Problema:

> Come aumentare la fedeltà a un soggetto specifico?

Soluzione:

fine-tuning del diffusion model + prior preservation.

Vantaggio:

maggiore subject fidelity.

Limite:

- più compute;
- più memoria;
- overfitting;
- un modello/adattamento per soggetto.

---

## ControlNet, 2023

Problema:

> Come controllare precisamente geometria e struttura dell'immagine?

Soluzione:

conditioning addizionale:

$$
\text{edge},
\text{depth},
\text{pose},
\text{segmentation},
\dots
$$

tramite ramo trainabile e zero convolution.

Limite:

- costo aggiuntivo;
- modelli specifici per differenti controlli;
- dipendenza dalla qualità del segnale strutturale.

---

## DiT, 2022/2023

Problema:

> La U-Net è veramente l'architettura migliore per scalare diffusion model sempre più grandi?

Soluzione:

$$
\text{U-Net}
\rightarrow
\text{Transformer}.
$$

Il latent viene patchificato e trattato come sequenza.

Vantaggio:

ottime proprietà di scaling.

Limite:

- costo dell'attention;
- necessità di molto compute;
- ridotto inductive bias rispetto alle CNN.

---

# 15. Le cinque equazioni da conoscere

Se devi sostenere un esame sui diffusion model, queste sono probabilmente le relazioni più importanti.

## 1. Forward transition

$$
q(x_t|x_{t-1})
=
\mathcal N
(
\sqrt{\alpha_t}x_{t-1},
(1-\alpha_t)I
).
$$

## 2. Closed-form forward process

$$
\boxed{
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon
}
$$

## 3. DDPM training loss

$$
\boxed{
L
=
\mathbb E
[
\|
\epsilon-
\epsilon_\theta(x_t,t)
\|^2
]
}
$$

## 4. Ricostruzione di $x_0$

$$
\boxed{
\hat x_0
=
\frac{
x_t-
\sqrt{1-\bar\alpha_t}
\epsilon_\theta(x_t,t)
}{
\sqrt{\bar\alpha_t}
}
}
$$

## 5. Latent diffusion

$$
\boxed{
z=E(x),
\qquad
\text{diffusion su }z,
\qquad
x=D(z)
}
$$

Se queste cinque formule sono chiare, gran parte della letteratura diventa molto più facile da leggere.

---

# 16. Forward diffusion, reverse diffusion e denoising: distinzione concettuale

Una fonte frequente di confusione è considerare il diffusion model semplicemente un denoising autoencoder.

Non è esattamente così.

Un denoising autoencoder classico impara:

$$
\tilde x\rightarrow x.
$$

Un diffusion model impara una famiglia di problemi:

$$
(x_t,t)
\rightarrow
\epsilon.
$$

Il timestep è fondamentale perché lo stesso input visivamente rumoroso deve essere interpretato rispetto al livello di rumore.

Possiamo considerare la rete come una funzione:

$$
f_\theta:
\mathcal X
\times
\{1,\dots,T\}
\rightarrow
\mathcal X.
$$

---

# 17. Perché la Gaussianità è così importante?

Il forward process utilizza gaussiane perché possiedono proprietà estremamente convenienti.

La composizione di trasformazioni gaussiane rimane gaussiana.

Questo è ciò che permette di passare direttamente da:

$$
x_0
$$

a:

$$
x_t.
$$

Se dovessimo realmente simulare:

$$
x_0\rightarrow x_1
\rightarrow\dots\rightarrow x_t
$$

durante ogni training step, il training sarebbe enormemente più costoso.

La relazione:

$$
q(x_t|x_0)
=
\mathcal N
(
\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I
)
$$

elimina completamente questo problema.

---

# 18. Interpretazione come Signal-to-Noise Ratio

Possiamo anche interpretare:

$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon
$$

in termini di signal-to-noise ratio.

La parte di segnale ha varianza proporzionale a:

$$
\bar\alpha_t.
$$

La parte di rumore:

$$
1-\bar\alpha_t.
$$

Una definizione comune è:

$$
\operatorname{SNR}(t)
=
\frac{
\bar\alpha_t
}{
1-\bar\alpha_t
}.
$$

Per $t$ piccolo:

$$
\operatorname{SNR}\gg1.
$$

Il segnale domina.

Per $t$ grande:

$$
\operatorname{SNR}\ll1.
$$

Il rumore domina.

Molti lavori successivi sui diffusion model possono essere compresi come tentativi di gestire meglio la distribuzione dell'SNR durante training e sampling.

---

# 19. Un modello mentale utile

È utile pensare alla generazione nel seguente modo.

A $t=T$:

> il modello vede quasi solo rumore e deve costruire una struttura globale.

A timestep intermedi:

> deve organizzare oggetti, forme e layout.

A timestep piccoli:

> deve correggere texture e dettagli.

Questa interpretazione non è una separazione matematica rigida, ma è utile per capire perché il denoising iterativo funzioni bene.

---

# 20. Confronto sintetico

| Metodo | Innovazione principale | Cosa viene appreso/modificato | Problema affrontato |
|---|---|---|---|
| Diffusion 2015 | Forward + reverse diffusion | Reverse process | Modellazione probabilistica |
| DDPM | Noise prediction | U-Net | Qualità e semplicità |
| DDIM | Sampling non-Markoviano | Nessun nuovo training | Velocità |
| iDDPM | Schedule + variance + loss | U-Net + variance | Likelihood/efficienza |
| LDM | Diffusion nei latent | U-Net latent + VAE | Costo ad alta risoluzione |
| Textual Inversion | Nuovo token | Embedding | Personalizzazione leggera |
| DreamBooth | Fine-tuning soggetto | Pesi diffusion | Subject fidelity |
| ControlNet | Conditioning spaziale | Ramo ControlNet | Controllo geometrico |
| DiT | Transformer backbone | Transformer | Scalabilità |

---

# 21. Collegamenti da ricordare per un esame

Il modo migliore di memorizzare i paper non è impararne i dettagli separatamente, ma ricordare la domanda a cui ciascuno risponde.

### Sohl-Dickstein

> Posso trasformare gradualmente una distribuzione complessa in una semplice e imparare il percorso inverso?

### DDPM

> Posso trasformare questo processo in un semplice problema di noise prediction?

### DDIM

> Devo veramente percorrere tutti i timestep?

### iDDPM

> Posso migliorare schedule, likelihood e varianza del reverse process?

### Latent Diffusion

> Devo veramente lavorare nello spazio dei pixel?

### Textual Inversion

> Posso insegnare un nuovo concetto modificando solamente il linguaggio?

### DreamBooth

> E se un embedding non fosse abbastanza potente per catturare l'identità?

### ControlNet

> Il testo dice cosa generare; come faccio a controllare dove e con quale struttura?

### DiT

> La U-Net è necessaria oppure possiamo applicare la filosofia di scaling dei Transformer?

---

# 22. La vera evoluzione dei diffusion model

L'evoluzione può essere riassunta lungo quattro assi indipendenti.

## Asse 1 — migliore modellazione

$$
\text{Diffusion 2015}
\rightarrow
\text{DDPM}
\rightarrow
\text{iDDPM}
$$

Domanda:

> come impariamo correttamente il processo generativo?

---

## Asse 2 — sampling più veloce

$$
\text{DDPM}
\rightarrow
\text{DDIM}
\rightarrow
\text{sampler moderni}.
$$

Domanda:

> come riduciamo il numero di neural function evaluations?

---

## Asse 3 — efficienza e scaling

$$
\text{pixel diffusion}
\rightarrow
\text{latent diffusion}
\rightarrow
\text{DiT}.
$$

Domanda:

> come aumentiamo capacità e risoluzione senza far esplodere il costo?

---

## Asse 4 — controllabilità

Si divide ulteriormente in:

### Controllo semantico

$$
\text{text conditioning}.
$$

### Personalizzazione

$$
\text{Textual Inversion}
\rightarrow
\text{DreamBooth}.
$$

### Controllo geometrico

$$
\text{ControlNet}.
$$

Questi ultimi tre non devono essere interpretati come sostituti diretti.

Per esempio è perfettamente possibile usare contemporaneamente:

$$
\text{Latent Diffusion}
+
\text{DreamBooth}
+
\text{ControlNet}
+
\text{DDIM sampler}.
$$

Ciascun componente risolve un problema differente.

---

# 23. Esempio finale: come si combinano tutte le idee

Supponiamo di voler generare:

> "il mio cane, seduto nella posa specificata da uno skeleton, in stile cinematografico."

Possiamo utilizzare:

### Latent Diffusion

per lavorare efficientemente in:

$$
z\text{-space}.
$$

### DreamBooth

per imparare l'identità:

$$
[V]\text{ dog}.
$$

### ControlNet

per imporre la posa:

$$
c=\text{pose map}.
$$

### Text conditioning

per specificare:

> "cinematic lighting".

### DDIM

per ridurre il numero di sampling step.

### Eventualmente un Transformer backbone

in una famiglia moderna di modelli derivata dalla filosofia DiT.

La distribuzione diventa concettualmente:

$$
p_\theta
(
z_0
\mid
\text{text},
\text{subject},
\text{pose}
).
$$

La generazione parte da:

$$
z_T\sim\mathcal N(0,I)
$$

e procede:

$$
z_T
\rightarrow
z_{t_1}
\rightarrow
z_{t_2}
\rightarrow
\dots
\rightarrow
z_0
$$

seguendo un sampler come DDIM.

Infine:

$$
x=D(z_0).
$$

Questa singola pipeline contiene praticamente tutta la linea evolutiva dei lavori considerati.