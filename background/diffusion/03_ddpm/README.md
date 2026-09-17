# Denoising Diffusion Probabilistic Models

I **Denoising Diffusion Probabilistic Models (DDPM)** definiscono un modello generativo a variabili latenti nel quale i livelli rumorosi $x_1,\ldots,x_T$ costituiscono variabili intermedie. Il processo forward è fissato e distrugge gradualmente l'informazione; il processo reverse è parametrizzato da una rete neurale e ricostruisce una distribuzione sui dati.

Il contributo fondamentale dei DDPM non consiste soltanto nell'aggiungere e rimuovere rumore. La costruzione permette di derivare un obiettivo variazionale trattabile e, grazie alla struttura gaussiana, di ridurlo a problemi di denoising supervisionato a differenti livelli di rumore.

## Forward process

La catena forward è

$$
q(x_{1:T}\mid x_0)
=
\prod_{t=1}^{T}
q(x_t\mid x_{t-1}),
$$

con

$$
q(x_t\mid x_{t-1})
=
\mathcal{N}
\left(
x_t;
\sqrt{\alpha_t}x_{t-1},
(1-\alpha_t)I
\right).
$$

Ponendo $\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$, una marginale può essere campionata in un solo passaggio:

$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

Il **noise schedule** determina la sequenza $\alpha_t$ e quindi la velocità con cui il segnale viene distrutto. Perché il prior finale sia effettivamente vicino a $\mathcal{N}(0,I)$, $\bar\alpha_T$ deve essere sufficientemente piccolo.

## Posteriore forward e processo reverse

La distribuzione $q(x_{t-1}\mid x_t)$ dipende dalla distribuzione ignota dei dati. Durante il training è però disponibile $x_0$, e il posteriore

$$
q(x_{t-1}\mid x_t,x_0)
$$

è gaussiano con media e varianza calcolabili in forma chiusa:

$$
q(x_{t-1}\mid x_t,x_0)
=
\mathcal{N}
\left(
x_{t-1};
\tilde\mu_t(x_t,x_0),
\tilde\beta_t I
\right),
$$

dove

$$
\tilde\beta_t
=
\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t.
$$

Il processo generativo appreso assume

$$
p_\theta(x_{0:T})
=
p(x_T)
\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t),
$$

con

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal{N}
\left(
x_{t-1};
\mu_\theta(x_t,t),
\Sigma_\theta(x_t,t)
\right).
$$

La rete può predire direttamente la media, ma è più comune parametrizzarla attraverso rumore, dato pulito o velocity. Con noise prediction,

$$
\mu_\theta(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-
\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t)
\right).
$$

## Evidence lower bound

La log-likelihood $\log p_\theta(x_0)$ contiene un'integrazione sulle variabili latenti. Introducendo il forward process come distribuzione variazionale si ottiene una **evidence lower bound (ELBO)**:

$$
\log p_\theta(x_0)
\geq
\mathbb{E}_{q(x_{1:T}\mid x_0)}
\left[
\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}
\right].
$$

La negative ELBO si decompone in un termine sul prior finale, una somma di divergenze KL tra posteriori forward e reverse appresi e un termine di ricostruzione per $x_0$. Grazie alle gaussiane, i termini intermedi diventano errori quadratici pesati.

Il [lavoro originale sui DDPM](https://arxiv.org/abs/2006.11239) osserva che una loss semplificata funziona particolarmente bene per la qualità dei campioni:

$$
\mathcal{L}_{\mathrm{simple}}
=
\mathbb{E}_{x_0,t,\epsilon}
\left[
\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|_2^2
\right].
$$

Questa loss non conserva i pesi esatti della ELBO. Ottimizzare likelihood e ottimizzare qualità percettiva non sono quindi obiettivi perfettamente equivalenti.

## Parametrizzazioni del target

Con

$$
x_t=\alpha(t)x_0+\sigma(t)\epsilon,
$$

le scelte più comuni sono:

- **$\epsilon$-prediction**, che stima il rumore;
- **$x_0$-prediction**, che ricostruisce direttamente il dato pulito;
- **$v$-prediction**, con $v=\alpha(t)\epsilon-\sigma(t)x_0$.

La prima è la parametrizzazione classica. La seconda offre un'interpretazione diretta ma può essere sensibile alle regioni a SNR molto basso. La $v$-prediction bilancia segnale e rumore lungo lo schedule ed è diffusa nei latent diffusion model. Poiché la MSE attribuisce peso uniforme nello spazio del target scelto, le tre versioni inducono dinamiche di ottimizzazione differenti.

## Sampling ancestrale

Il sampler DDPM parte da $x_T\sim\mathcal{N}(0,I)$ e, per $t=T,\ldots,1$, campiona

$$
x_{t-1}
=
\mu_\theta(x_t,t)
+
\sigma_t z,
\qquad
z\sim\mathcal{N}(0,I),
$$

omettendo il rumore aggiuntivo all'ultimo passo. Il termine casuale rende il sampler **ancestrale**: ogni transizione produce un nuovo campione condizionato sullo stato corrente.

Usare tutti i timestep di training può richiedere centinaia o migliaia di valutazioni. Sampler successivi riducono il numero di passi selezionando una griglia più rada o integrando una dinamica continua con metodi numerici più accurati.

## DDIM e sampling deterministico

I **Denoising Diffusion Implicit Models (DDIM)** mantengono le stesse marginali forward e lo stesso obiettivo di training, ma definiscono un processo generativo non-Markoviano. Un parametro $\eta$ controlla la stochasticity: con $\eta=0$ il percorso è deterministico una volta fissato $x_T$.

DDIM consente di saltare timestep e ridurre il numero di valutazioni, spesso con un compromesso favorevole tra velocità e qualità. Non trasforma però automaticamente qualsiasi modello in un generatore one-step: riduzioni aggressive amplificano l'errore numerico e l'errore del denoiser.

## Improved DDPM

Il lavoro **Improved DDPM** separa alcuni limiti della configurazione originaria. Introduce uno schedule cosine per evitare che l'informazione venga distrutta troppo rapidamente, apprende la varianza del reverse process entro un intervallo definito dalle varianze forward e combina la loss semplificata con un piccolo termine variazionale.

La learned variance non sostituisce la predizione della media. Il modello produce anche un coefficiente che interpola, in log-space, tra due scelte di varianza analitiche. Il termine di ELBO associato alla varianza viene ottimizzato senza lasciare che i suoi gradienti modifichino la stima della media, stabilizzando la loss ibrida.

Per ottimizzare direttamente la likelihood, i timestep possono essere campionati con importance sampling in proporzione alla difficoltà stimata dei rispettivi termini. Queste modifiche migliorano likelihood e sampling con meno passi, ma confermano che **schedule, mean parameterization e reverse variance sono componenti distinte**. Il riferimento è [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672).

## Architettura del denoiser

Il DDPM non prescrive una singola architettura. Storicamente viene usata una **U-Net** con blocchi residuali, attenzione alle risoluzioni più basse e skip connection tra encoder e decoder. Il timestep viene codificato con embedding sinusoidali o Fourier e iniettato nei blocchi, spesso tramite trasformazioni affine o adaptive normalization.

Il denoiser deve conoscere il livello di rumore perché lo stesso $x_t$ richiede interpretazioni differenti a SNR differenti. Nei modelli condizionati, la condizione entra attraverso concatenazione, feature modulation oppure cross-attention. I moderni Diffusion Transformer sostituiscono la U-Net convoluzionale con blocchi Transformer senza cambiare necessariamente l'obiettivo DDPM.

## Classifier-free guidance

La **classifier-free guidance (CFG)** addestra lo stesso modello sia con sia senza condizione, rimuovendo casualmente $c$ durante il training. In inferenza combina le due predizioni:

$$
\hat\epsilon_{\mathrm{CFG}}
=
\hat\epsilon_\theta(x_t,t,\varnothing)
+
w
\left[
\hat\epsilon_\theta(x_t,t,c)
-
\hat\epsilon_\theta(x_t,t,\varnothing)
\right],
$$

dove $w$ è la guidance scale. Aumentare $w$ rafforza l'aderenza alla condizione, ma può ridurre diversità, saturare il campione e portarlo fuori dalla distribuzione vista in training.

## Punti di forza e limiti

I DDPM offrono una loss stabile, un obiettivo supervisionato semplice e una capacità elevata di modellare distribuzioni multimodali. Separano inoltre l'addestramento del denoiser dalla scelta del sampler, consentendo di migliorare l'inferenza senza riaddestrare sempre il modello.

Il limite principale è il **sampling iterativo**. Il costo cresce con il numero di valutazioni del network e diventa critico nel controllo robotico in tempo reale. La qualità dipende anche da schedule, parametrizzazione, loss weighting e solver: indicare soltanto l'architettura non è sufficiente a rendere riproducibile un risultato.

Il riferimento fondamentale è [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239); il sampling implicito è introdotto in [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502).
