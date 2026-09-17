# Score-Based Models, SDE e diffusione continua

I **score-based generative model** apprendono il campo

$$
s_t(x)=\nabla_x\log p_t(x),
$$

cioè il gradiente della log-densità della distribuzione al tempo $t$. Questa prospettiva sposta l'attenzione dalla singola transizione DDPM all'intera famiglia di distribuzioni rumorose. La formulazione tramite **stochastic differential equations (SDE)** mostra poi che diffusion model discreti e precedenti metodi score-based sono casi particolari di uno stesso processo continuo.

## Perché apprendere lo score

Stimare direttamente una densità normalizzata in alta dimensione è difficile. Lo score non dipende invece dalla costante di normalizzazione:

$$
\nabla_x\log\frac{\tilde p(x)}{Z}
=
\nabla_x\log\tilde p(x).
$$

Se fosse noto, potrebbe guidare campioni da regioni a bassa densità verso regioni più probabili. Tuttavia, lo score dei dati può essere mal definito lontano dalla manifold e non è disponibile come target supervisionato.

Aggiungere rumore gaussiano risolve entrambe le difficoltà. Le distribuzioni perturbate $p_t$ sono più lisce e il target condizionale è calcolabile. Per

$$
x_t=\alpha(t)x_0+\sigma(t)\epsilon
$$

si ha

$$
\nabla_{x_t}\log p(x_t\mid x_0)
=
-\frac{\epsilon}{\sigma(t)}.
$$

## Denoising score matching

Il **denoising score matching** minimizza

$$
\mathcal{L}_{\mathrm{DSM}}(\theta)
=
\mathbb{E}_{t,x_0,x_t}
\left[
\lambda(t)
\left\|
s_\theta(x_t,t)
-
\nabla_{x_t}\log p(x_t\mid x_0)
\right\|_2^2
\right].
$$

Il simbolo $\lambda(t)$ indica un peso scelto per bilanciare i livelli di rumore. Anche se il target usa $x_0$, il minimizzatore medio coincide con lo score marginale $\nabla_{x_t}\log p_t(x_t)$. In questo senso, la noise prediction dei DDPM è una particolare parametrizzazione dello score matching.

## Forward SDE

Una SDE di Itô descrive il processo forward continuo:

$$
d x
=
f(x,t)\,dt
+
g(t)\,dW_t,
$$

dove $f$ è il **drift**, $g$ l'ampiezza della diffusione e $W_t$ un processo di Wiener. Partendo da $x(0)\sim p_{\mathrm{data}}$, i coefficienti sono scelti affinché $x(T)$ segua approssimativamente una distribuzione semplice.

L'SDE induce una famiglia di marginali $p_t(x)$. La loro evoluzione è descritta dall'equazione di Fokker–Planck:

$$
\partial_t p_t
=
-\nabla\cdot(fp_t)
+
\frac{1}{2}g(t)^2\Delta p_t.
$$

Il primo termine trasporta la densità; il secondo la diffonde. La descrizione continua separa la dinamica teorica dalla griglia temporale usata nell'implementazione.

## Reverse-time SDE

Il risultato centrale è che il processo inverso è ancora una SDE:

$$
d x
=
\left[
f(x,t)
-
g(t)^2\nabla_x\log p_t(x)
\right]dt
+
g(t)\,d\bar W_t,
$$

dove il tempo viene integrato da $T$ a $0$ e $\bar W_t$ è un processo di Wiener nel verso inverso. Sostituendo lo score ignoto con $s_\theta(x,t)$ si ottiene una dinamica campionabile.

La componente stocastica non è un rumore aggiunto arbitrariamente al sampler: è parte della dinamica inversa che riproduce le marginali corrette quando lo score è esatto.

## Probability flow ODE

La stessa famiglia di marginali può essere prodotta da una ODE deterministica:

$$
\frac{dx}{dt}
=
f(x,t)
-
\frac{1}{2}g(t)^2
\nabla_x\log p_t(x).
$$

Questa **probability flow ODE** condivide $p_t$ con la SDE, ma non le stesse traiettorie individuali. Una volta fissata la condizione iniziale, il percorso è deterministico. L'ODE permette di usare solver standard e, in principio, calcolare likelihood mediante la divergenza del campo.

SDE inversa e probability flow ODE non devono essere confuse con due modelli differenti. Sono due dinamiche generative costruite a partire dallo stesso score.

## Variance Preserving SDE

La **Variance Preserving (VP) SDE** è il limite continuo del forward process DDPM:

$$
d x
=
-\frac{1}{2}\beta(t)x\,dt
+
\sqrt{\beta(t)}\,dW_t.
$$

La varianza totale rimane controllata mentre il segnale decade. Discretizzando opportunamente questa SDE si recuperano transizioni del tipo

$$
x_t
=
\sqrt{1-\beta_t}x_{t-1}
+
\sqrt{\beta_t}\epsilon.
$$

Il DDPM è quindi interpretabile come una particolare discretizzazione temporale di una VP SDE, non come una famiglia concettualmente separata.

## Variance Exploding e sub-VP SDE

La **Variance Exploding (VE) SDE** mantiene nullo il drift e aumenta progressivamente la scala del rumore:

$$
d x=g(t)\,dW_t.
$$

Le perturbazioni hanno varianza crescente senza attenuare esplicitamente il dato. Questa formulazione collega le SDE ai Noise Conditional Score Networks, che apprendono score su molte scale di rumore.

La **sub-VP SDE** modifica il coefficiente di diffusione per ottenere proprietà favorevoli nella likelihood. VP, VE e sub-VP condividono il principio generale, ma generano differenti marginali e richiedono coefficienti coerenti nel reverse process.

## Predictor-corrector sampling

Il framework score-based consente di combinare due aggiornamenti. Il **predictor** discretizza la reverse-time SDE, per esempio con Euler–Maruyama. Il **corrector** applica passi di Langevin dynamics usando lo score per correggere il campione alla distribuzione marginale corrente.

Schematicamente:

```text
predictor: avanzamento verso un livello di rumore inferiore
corrector: raffinamento alla distribuzione del livello corrente
```

Più correzioni possono migliorare la fedeltà ma aumentano le valutazioni del network. Solver ODE e metodi multistep offrono alternative deterministiche; la scelta dipende dal budget di calcolo e dalla robustezza dello score fuori distribuzione.

## Unificazione di diffusione discreta e continua

Il quadro SDE chiarisce le corrispondenze:

- lo schedule DDPM diventa il coefficiente temporale di una VP SDE;
- la noise prediction viene convertita nello score tramite $s_\theta=-\epsilon_\theta/\sigma(t)$;
- il reverse sampler è una discretizzazione della reverse-time SDE;
- DDIM è collegato a una dinamica deterministica affine alla probability flow ODE;
- differenti sampler corrispondono a differenti discretizzazioni, non necessariamente a differenti modelli addestrati.

Questa separazione permette di studiare indipendentemente probability path, parametrizzazione del network e metodo numerico. Consente inoltre di trasferire solver sviluppati per ODE e SDE al sampling generativo.

## Limiti pratici

Le equivalenze teoriche assumono uno score esatto e integrazione infinitesimale. Nella pratica il network ha errore, la griglia contiene pochi passi e la guidance può amplificare regioni non calibrate. Due solver che condividono lo stesso limite continuo possono quindi produrre qualità e stabilità differenti.

Lo score può inoltre divergere quando $\sigma(t)\rightarrow0$, rendendo delicati gli estremi temporali. Preconditioning, parametrizzazione del target e scelta dei limiti $t_{\min}$ e $t_{\max}$ sono parte integrante del sistema.

Il riferimento principale è [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456), che formalizza reverse-time SDE, probability flow ODE e predictor-corrector sampling in un quadro unitario.
