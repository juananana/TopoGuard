%% TopoGuard: Adaptive Topology Orchestration for Risk-Sensitive Decision
%% under Hard Constraints and Multimodal Uncertainty
%% Submitted to ACM Transactions on Multimedia Computing, Communications and Applications (TOMM)
%%
%% Commands for TeXCount
%TC:macro \cite [option:text,text]
%TC:macro \citep [option:text,text]
%TC:macro \citet [option:text,text]
%TC:envir table 0 1
%TC:envir table* 0 1
%TC:envir tabular [ignore] word
%TC:envir displaymath 0 word
%TC:envir math 0 word
%TC:envir comment 0 0
%%
\documentclass[manuscript,review,anonymous]{acmart}

\usepackage{graphicx}
\usepackage{algorithm}
\usepackage{algpseudocode}

\setcopyright{none}
\settopmatter{printacmref=true}
\renewcommand\footnotetextcopyrightpermission[1]{}


%%
%% \BibTeX command to typeset BibTeX logo in the docs
\AtBeginDocument{%
  \providecommand\BibTeX{{%
    Bib\TeX}}}

%% Rights management information.  This information is sent to you
%% when you complete the rights form.  These commands have SAMPLE
%% values in them; it is your responsibility as an author to replace
%% the commands and values with those provided to you when you
% %% complete the rights form.


% \setcopyright{acmlicensed}
% \copyrightyear{2018}
% \acmYear{2018}
% \acmDOI{XXXXXXX.XXXXXXX}
% %% These commands are for a PROCEEDINGS abstract or paper.
% \acmConference[Conference acronym 'XX]{Make sure to enter the correct
%   conference title from your rights confirmation email}{June 03--05,
%   2018}{Woodstock, NY}
%%
%%  Uncomment \acmBooktitle if the title of the proceedings is different
%%  from ``Proceedings of ...''!
%%
%%\acmBooktitle{Woodstock '18: ACM Symposium on Neural Gaze Detection,
%%  June 03--05, 2018, Woodstock, NY}
% \acmISBN{978-1-4503-XXXX-X/2018/06}


%%
%% Submission ID.
%% Use this when submitting an article to a sponsored event. You'll
%% receive a unique submission ID from the organizers
%% of the event, and this ID should be used as the parameter to this command.
%%\acmSubmissionID{123-A56-BU3}

%%
%% For managing citations, it is recommended to use bibliography
%% files in BibTeX format.
%%
%% You can then either use BibTeX with the ACM-Reference-Format style,
%% or BibLaTeX with the acmnumeric or acmauthoryear sytles, that include
%% support for advanced citation of software artefact from the
%% biblatex-software package, also separately available on CTAN.
%%
%% Look at the sample-*-biblatex.tex files for templates showcasing
%% the biblatex styles.
%%

%%
%% The majority of ACM publications use numbered citations and
%% references.  The command \citestyle{authoryear} switches to the
%% "author year" style.
%%
%% If you are preparing content for an event
%% sponsored by ACM SIGGRAPH, you must use the "author year" style of
%% citations and references.
%% Uncommenting
%% the next command will enable that style.
%%\citestyle{acmauthoryear}


%%
%% end of the preamble, start of the body of the document source.
\begin{document}

%%
%% The "title" command has an optional parameter,
%% allowing the author to define a "short title" to be used in page headers.
\title{TopoGuard: Adaptive Topology Orchestration for Risk-Sensitive Decision  under Hard Constraints and Multimodal Uncertainty }

%%
%% The "author" command and its associated commands are used to define
%% the authors and their affiliations.
%% Of note is the shared affiliation of the first two authors, and the
%% "authornote" and "authornotemark" commands
%% used to denote shared contribution to the research.

 

% \author{Ben Trovato}
% \authornote{Both authors contributed equally to this research.}
% \email{trovato@corporation.com}
% \orcid{1234-5678-9012}
% \author{G.K.M. Tobin}
% \authornotemark[1]
% \email{webmaster@marysville-ohio.com}
% \affiliation{%
%   \institution{Institute for Clarity in Documentation}
%   \city{Dublin}
%   \state{Ohio}
%   \country{USA}
% }

% \author{Lars Th{\o}rv{\"a}ld}
% \affiliation{%
%   \institution{The Th{\o}rv{\"a}ld Group}
%   \city{Hekla}
%   \country{Iceland}}
% \email{larst@affiliation.org}

% \author{Valerie B\'eranger}
% \affiliation{%
%   \institution{Inria Paris-Rocquencourt}
%   \city{Rocquencourt}
%   \country{France}
% }

% \author{Aparna Patel}
% \affiliation{%
%  \institution{Rajiv Gandhi University}
%  \city{Doimukh}
%  \state{Arunachal Pradesh}
%  \country{India}}

% \author{Huifen Chan}
% \affiliation{%
%   \institution{Tsinghua University}
%   \city{Haidian Qu}
%   \state{Beijing Shi}
%   \country{China}}

% \author{Charles Palmer}
% \affiliation{%
%   \institution{Palmer Research Laboratories}
%   \city{San Antonio}
%   \state{Texas}
%   \country{USA}}
% \email{cpalmer@prl.com}

% \author{John Smith}
% \affiliation{%
%   \institution{The Th{\o}rv{\"a}ld Group}
%   \city{Hekla}
%   \country{Iceland}}
% \email{jsmith@affiliation.org}

% \author{Julius P. Kumquat}
% \affiliation{%
%   \institution{The Kumquat Consortium}
%   \city{New York}
%   \country{USA}}
% \email{jpkumquat@consortium.net}



%%
%% By default, the full list of authors will be used in the page
%% headers. Often, this list is too long, and will overlap
%% other information printed in the page headers. This command allows
%% the author to define a more concise list
%% of authors' names for this purpose.



% \renewcommand{\shortauthors}{Trovato et al.}



%%
%% The abstract is a short summary of the work to be presented in the
%% article.
\begin{abstract}
Risk-sensitive decision systems, such as storm-surge warning and emergency response, often depend on multi-stage collaboration among heterogeneous modules and multimodal evidence. However, such systems face a dual challenge: multimodal uncertainty (e.g., noisy sensors or conflicting weather patterns) destabilizes evidence reliability, while hard constraints (e.g., safety thresholds, strict time windows, and mandatory human oversight) prohibit unconstrained online exploration. Fixed pipelines are brittle under uncertainty, yet trial-and-error is infeasible under strict safety boundaries. We present \textbf{TopoGuard}, an adaptive topology orchestration framework that constructs feasible and reliable execution workflows through profile-guided topology selection and bounded local repair. Instead of exhaustive online search, TopoGuard maintains task-conditional Accuracy--Cost profiles for candidate agent combinations, which guide the generation of an initial high-quality topology aligned with current risk levels and constraints. During execution, evaluator modules monitor intermediate results and trigger local repair when failures occur, upgrading only affected stages to ensure robustness without full replanning. Experiments on a water-conservancy digital-twin platform demonstrate that TopoGuard significantly outperforms fixed pipelines in quality while maintaining favorable cost-latency trade-offs, and that bounded local repair provides effective recovery for difficult execution cases. These results indicate that profile-guided topology selection with bounded local repair effectively navigates the quality-cost-latency trade-off under hard operational constraints for constrained multimodal decision systems.
\end abstract}

%%
%% The code below is generated by the tool at http://dl.acm.org/ccs.cfm.
%% Please copy and paste the code instead of the example below.
%%
\begin{CCSXML}
<ccs2012>
   <concept>
       <concept_id>10002951.10003227.10003241</concept_id>
       <concept_desc>Information systems~Decision support systems</concept_desc>
       <concept_significance>500</concept_significance>
   </concept>
   <concept>
       <concept_id>10002951.10003227.10003251</concept_id>
       <concept_desc>Information systems~Multimedia information systems</concept_desc>
       <concept_significance>500</concept_significance>
   </concept>
   <concept>
       <concept_id>10010147.10010178</concept_id>
       <concept_desc>Computing methodologies~Artificial intelligence</concept_desc>
       <concept_significance>300</concept_significance>
   </concept>
   <concept>
       <concept_id>10010147.10010178.10010199</concept_id>
       <concept_desc>Computing methodologies~Planning and scheduling</concept_desc>
       <concept_significance>300</concept_significance>
   </concept>
</ccs2012>
\end{CCSXML}

\ccsdesc[500]{Information systems~Decision support systems}
\ccsdesc[500]{Information systems~Multimedia information systems}
\ccsdesc[300]{Computing methodologies~Artificial intelligence}
\ccsdesc[300]{Computing methodologies~Planning and scheduling}

%%
%% Keywords. The author(s) should pick words that accurately describe
%% the work being presented. Separate the keywords with commas.
\keywords{Multimedia Systems, Topology Orchestration, Multimodal Decision Support, Workflow Optimization, Local Graph Repair, Digital Twin}
%% A "teaser" image appears between the author and affiliation
%% information and the body of the document, and typically spans the
%% page.

\begin{teaserfigure}
\centering
\includegraphics[width=\linewidth]{outputs/overall_water_qa/figures/fig2_v2.pdf}
\caption{Overview of the TopoGuard framework. The system takes a task description, resource constraints, and multimodal inputs, constructs an initial executable workflow through topology-level and executor-level Pareto-guided selection, monitors execution with an evaluator, and performs bounded local recovery through executor upgrade, verifier upgrade, or topology refinement when failures occur.}
\Description{Overview of the TopoGuard framework. The system takes a task description, resource constraints, and multimodal inputs, constructs an initial executable workflow through topology-level and executor-level Pareto-guided selection, monitors execution with an evaluator, and performs bounded local recovery through executor upgrade, verifier upgrade, or topology refinement when failures occur.}
\label{fig:framework}
\end{teaserfigure}

% \received{20 February 2007}
% \received[revised]{12 March 2009}
% \received[accepted]{5 June 2009}

%%
%% This command processes the author and affiliation and title
%% information and builds the first part of the formatted document.
\maketitle

\section{Introduction}
Modern multimodal intelligent systems are increasingly used for decision support in risk-sensitive domains such as hydrological analysis, infrastructure monitoring, and emergency response \cite{algiriyage2022multi,cheng2023review,inyang2025digital,tomczak2024development}. In these settings, task completion typically requires the coordinated use of heterogeneous functional modules, including data retrieval, numerical computation, reasoning, and verification \cite{zhang2024mm,huang2024understanding,zhai2025survey}. System effectiveness therefore depends not only on individual executors, but also on how they are organized into an executable collaboration topology.

However, such systems face a dual challenge: multimodal uncertainty (e.g., noisy sensors or conflicting weather patterns) destabilizes evidence reliability, while hard constraints (e.g., safety thresholds, strict time windows, and mandatory human oversight) prohibit unconstrained online exploration. Many deployed systems still rely on manually designed workflows or fixed model-selection rules where functional nodes and execution order are predetermined \cite{zeng2023flowmind,li2024autoflow,zhang2024aflow,fan2024workflowllm}. Such designs are easy to deploy but brittle in uncertain environments: different task instances require different processing structures; intermediate outputs may fail; and execution must satisfy constraints on quality, cost, and latency. A workflow that performs well for one instance may become expensive, slow, or ineffective for another, especially in risk-sensitive applications where local failures can propagate into unreliable downstream decisions \cite{tang2016emergency}. This issue is further amplified in multimodal settings with missing or unreliable modalities \cite{wu2024deep,lin2023missmodal,lan2024robust,reza2024robust,liaqat2025chameleon}.

Existing approaches attempt to improve adaptability through model routing or dynamic tool selection \cite{schick2023toolformer,yao2023react,kim2024llm}. However, the core decision is not only which executor to use, but also which functional nodes should be activated, how they should be connected, and how the execution graph should be adjusted when failures occur. In other words, the decision variable is the workflow topology itself. Recent work on LLM planning and workflow generation highlights structured execution organization, but mainly focuses on planning quality rather than topology adaptation under runtime constraints \cite{huang2024understanding,zhai2025survey,li2024autoflow,zhang2024aflow,fan2024workflowllm}. We therefore formulate the problem as \emph{adaptive topology orchestration} under hard constraints and multimodal uncertainty.

To address this problem, we propose \textbf{TopoGuard}, an adaptive topology orchestration framework for risk-sensitive decision systems. TopoGuard treats workflow execution as a closed-loop decision process over structured graphs. It first builds offline performance profiles for candidate executors and estimates the expected quality, cost, and latency of candidate topologies. Based on these estimates, TopoGuard performs constrained Pareto-based selection to obtain an initial executable graph. During execution, evaluator modules monitor intermediate outputs and detect local failures or reliability risks. Instead of triggering global replanning, TopoGuard performs bounded local graph repair to improve robustness while controlling additional overhead.

The key insight of this work is that robust multimodal decision systems should move beyond choosing better components to choosing and maintaining better \emph{collaboration topologies}. Topology-level adaptation allows the execution graph to adjust to task requirements, resource constraints, and runtime uncertainty, which is essential for balancing effectiveness, efficiency, and reliability in risk-sensitive environments.

We evaluate TopoGuard on representative tasks involving heterogeneous executors and structured workflow operators. Experimental results show that TopoGuard achieves better quality-cost-latency trade-offs and stronger robustness than static baselines, particularly when task requirements vary and intermediate failures occur.

\noindent\textbf{The main contributions of this paper are summarized as follows:}
\begin{itemize}
    \item \textbf{Problem Formulation.} We formulate adaptive workflow execution in risk-sensitive multimodal systems as an adaptive topology orchestration problem under hard constraints and multimodal uncertainty.
    \item \textbf{TopoGuard Framework.} We propose \textbf{TopoGuard}, a feasibility-aware topology orchestration framework that integrates offline performance profiling, constrained Pareto-based initial topology selection, and evaluator-triggered bounded local repair.
    \item \textbf{Empirical Validation.} We demonstrate that TopoGuard achieves better quality-cost-latency trade-offs and stronger robustness than static and non-adaptive alternatives in constrained multimodal environments.
\end{itemize}

\section{Related Work}
\label{sec:related_work}

\subsection{Multimedia Decision Support and Digital Twins in High-Stakes Domains}

Multimedia systems are increasingly used beyond perception and retrieval, and have become part of decision-support pipelines in high-stakes settings such as emergency response, infrastructure monitoring, and hydrological analysis \cite{tang2016emergency,pathak2020hydroinformatics,yu2024warenet}. In water and infrastructure domains, recent digital-twin studies emphasize real-time monitoring, simulation, and decision support as core capabilities of intelligent management systems \cite{li2024intelligentwater,ge2025urbanflooddt}. These systems typically integrate heterogeneous observations, simulation outputs, and operational knowledge, but their execution logic is often implemented as manually designed pipelines or domain-specific control flows. Our work is related to this line of research in that it also targets decision support under multimodal inputs and operational constraints. The difference is that we focus on \emph{how} the execution structure should be selected and adjusted at runtime when task requirements, resource limits, and intermediate reliability vary across instances.

\subsection{Multimodal Uncertainty and Missing-Modality Robustness}

A large body of multimodal research studies robustness under missing, noisy, or unreliable inputs \cite{lan2025robust,wu2024mlmmsurvey}. This literature is directly relevant because real-world decision systems often operate with partial observability, degraded sensors, or cross-source inconsistency. Existing methods mainly improve robustness within a fixed model architecture or a fixed processing pipeline, for example through missing-modality learning, representation completion, or robust fusion. By contrast, we examine a different question: when evidence quality changes, should the system still use the same execution structure, or should it switch to a different workflow topology with additional verification or recovery stages? In this sense, our work complements model-level robustness methods by addressing structure-level adaptation.

\subsection{Workflow and Agent Orchestration}

Recent work on LLM-based agents has highlighted the value of structured execution rather than single-shot prompting. ReAct and Tree of Thoughts illustrate how intermediate reasoning and tool interaction can improve complex task solving \cite{yao2023react,yao2023tot}. More recent workflow-oriented approaches explicitly treat agent execution as a graph or workflow construction problem. For example, AFlow formulates agentic workflow generation as search over code-represented workflows and optimizes node-edge structure automatically \cite{zhang2024aflow}. These studies show that execution structure matters, but they primarily focus on producing strong workflows or better reasoning performance. Our setting is different in two respects: we consider hard deployment constraints on cost and latency during selection, and we explicitly model bounded local repair after execution has started.

\subsection{Cost-Aware Routing, Cascading, and Runtime Recovery}

Another closely related line of research studies cost-aware model selection and cascading. FrugalGPT shows that routing or cascading across models can reduce inference cost while maintaining strong task performance \cite{chen2023frugalgpt}. Subsequent work further unifies routing and cascading and highlights the importance of quality estimation for cost-performance trade-offs \cite{dekoninck2024routing}. These methods are relevant because they also optimize quality under budget constraints, but they usually treat the decision variable as \emph{which model} to call for a given query. Our work instead treats the workflow topology itself as the decision variable, jointly considering node activation, graph structure, executor assignment, and local repair.

Our repair component is also related to the broader literature on workflow exception handling and adaptive failure recovery. Prior research in workflow systems and service composition has studied exception handling, ad-hoc recovery, and adaptive failure handling for composite processes \cite{chiu2001exception,sato2010adaff}. Those studies motivate the importance of runtime recovery, but they are generally not formulated around modern multimodal agent workflows or multi-objective quality-cost-latency trade-offs. We build on the same general intuition that failures should be handled locally when possible, while instantiating it in a profile-guided orchestration framework for multimodal decision systems.




% Overleaf-ready English method section for direct paste.
% Suggested packages in your preamble:
% \usepackage{amsmath,amssymb,amsfonts}
% \usepackage{algorithm}
% \usepackage{algpseudocode}





\section{Method}
\label{sec:method}

\subsection{Problem Setup}

We consider adaptive orchestration for multi-stage decision tasks. For an input task $X$, the system must choose (i) a workflow structure, (ii) executor assignments for the nodes in that structure, and (iii) a limited set of repair actions if execution quality falls below a threshold.

The framework has two coupled stages: initial topology selection under hard quality--cost--latency constraints, followed by bounded local repair when intermediate failures are detected. Because the workflow-graph space is combinatorial, direct search is impractical; we therefore use a hierarchical selection procedure together with minimal-change local repair.

\subsection{Initial Topology Selection under Hard Constraints}
\label{sec:init_objective}

We first formulate the construction of the initial executable workflow as a constrained topology selection problem.
Given an input task $X$, the policy $\pi$ selects an initial workflow graph $G_{\pi}(X)$ from the candidate graph space.
The objective is to maximize utility under hard structural and resource constraints:
\begin{equation}
\pi^\star
=
\arg\max_{\pi\in\Pi}
\;\mathbb{E}_{X\sim D}\big[Q(G_{\pi}(X),X)\big],
\label{eq:init_obj}
\end{equation}
subject to
\begin{equation}
\begin{aligned}
&G_{\pi}(X)\in\mathcal{G}_{\mathrm{DAG}},\\
&C(G_{\pi}(X),X)\le B(X),\\
&L(G_{\pi}(X),X)\le \Delta(X).
\end{aligned}
\label{eq:init_constraints}
\end{equation}

Here, $\mathcal{G}_{\mathrm{DAG}}$ denotes the space of valid directed acyclic workflow graphs,
$B(X)$ is the task-specific budget constraint, and $\Delta(X)$ is the latency constraint.
This stage determines only the \emph{initial} executable topology; repair actions during execution are formulated separately in Section~\ref{sec:repair}.

To compare quality, cost, and latency on a comparable scale, we use a normalized utility:
\begin{equation}
Q(G,X)=\alpha\,S_{\mathrm{norm}}(G,X)-\beta\,C_{\mathrm{norm}}(G,X)-\gamma\,L_{\mathrm{norm}}(G,X),
\label{eq:utility}
\end{equation}
where $\alpha,\beta,\gamma\ge 0$ and $\alpha+\beta+\gamma=1$.
In our implementation, quality is normalized as
\begin{equation}
S_{\mathrm{norm}}(G,X)=\frac{S(G,X)}{S_{\mathrm{scale}}},
\label{eq:s_norm}
\end{equation}
and cost and latency use the normalized profile values $C_{\mathrm{norm}}$ and $L_{\mathrm{norm}}$ for candidate comparison.
Unless otherwise specified, we use $\alpha=0.65$, $\beta=0.25$, $\gamma=0.10$, and $S_{\mathrm{scale}}=1.5$.
Pareto frontiers are used only for candidate pruning before the final utility-based selection.

\subsection{Workflow Representation}

A workflow is represented as a directed graph
\begin{equation}
G=(V,E,\tau,\rho,\phi),
\end{equation}
where $V$ is the node set, $E\subseteq V\times V$ is the directed edge set, $\tau(v)$ is node role type, $\rho(v)$ is executor function type, and $\phi(v)$ is runtime configuration. DAG topology is enforced by a topological order $\operatorname{ord}:V\rightarrow\{1,\dots,|V|\}$ such that
\begin{equation}
(u,v)\in E\Rightarrow \operatorname{ord}(u)<\operatorname{ord}(v).
\end{equation}

Task context is encoded into a discrete difficulty level $b \in \{\text{easy}, \text{medium}, \text{hard}\}$, estimated from task features such as query length, domain complexity, and retrieval scope. The resulting difficulty level is used to condition both template retrieval and executor candidate selection.

\subsection{Difficulty-Aware Hierarchical Initialization}

Given task $X$ and difficulty level $b$, initialization is decomposed into two levels:
\begin{equation}
T_0=\pi_{\text{init}}^{T}(X,b),
\qquad
\Phi_0=\pi_{\text{init}}^{N}(X,b,T_0),
\qquad
G_0=(T_0,\Phi_0).
\end{equation}

At the template level, we first perform Pareto pruning over candidate topologies:
\begin{equation}
\mathcal{P}_T=\text{Pareto}_{\max S,\min C,\min L}(\mathcal{T}(X,b)),
\end{equation}
and then select the feasible template with highest proxy utility:
\begin{equation}
T_0=\arg\max_{T\in\mathcal{P}_T^{\text{feasible}}}\widehat{Q}_T(T;X,b).
\end{equation}

At the node level, each executor node retrieves a difficulty-conditioned candidate pool,
\begin{equation}
\mathcal{A}_v(b,\rho(v)),
\qquad
\mathcal{P}_N(v)=\text{Pareto}_{\max S,\min C}(\mathcal{A}_v(b,\rho(v))),
\end{equation}
and selects the feasible candidate with maximal utility:
\begin{equation}
Q_N(a;v,X,b)=\alpha_N S(a;v,X,b)-\beta_N C(a;v,X,b),
\end{equation}
\begin{equation}
\phi(v)=\arg\max_{a\in\mathcal{P}_N^{\text{feasible}}(v)}Q_N(a;v,X,b).
\end{equation}

\subsection{Bounded Local Repair with Edit-Loss}
\label{sec:repair}

After the initial workflow $G_0$ has been selected and instantiated, execution proceeds in a closed loop.
When the evaluator detects that an intermediate result fails the pass condition, TopoGuard does not re-optimize the whole graph from scratch.
Instead, it performs bounded local repair around the affected region.

At execution step $t$, let $G_t$ denote the current workflow graph and $s_t$ the local execution state.
A repair action is selected as
\begin{equation}
a_t=\pi_{\mathrm{repair}}(s_t),
\qquad
G_{t+1}=\mathcal{T}_{\mathrm{op}}(G_t,a_t),
\label{eq:repair_transition}
\end{equation}
where $\mathcal{T}_{\mathrm{op}}$ applies a local graph edit to the current workflow.

We consider three repair operators,
\begin{equation}
\mathcal{A}_t=\{A,B,C\},
\end{equation}
corresponding to topology/template upgrade, executor upgrade, and evaluator upgrade, respectively.
To discourage unnecessary large edits, we define the edit loss as
\begin{equation}
\mathcal{L}_{\mathrm{edit}}(a)
=
\lambda_n\Delta n(a)+\lambda_\phi\Delta\phi(a)+\lambda_e\Delta e(a),
\label{eq:edit_loss}
\end{equation}
where $\Delta n(a)$, $\Delta\phi(a)$, and $\Delta e(a)$ measure changes in nodes, executor assignments, and edges.
In our implementation, the weights are set to $\lambda_n=0.30$, $\lambda_\phi=0.25$, $\lambda_e=0.10$, and the overall penalty weight is $\lambda=0.20$.

The repair policy then solves a bounded local optimization problem:
\begin{equation}
a_t^\star
=
\arg\max_{a\in\mathcal{A}_t^{\mathrm{feasible}}}
\Big(
Q(G_{t+1}(a),X)-\lambda\,\mathcal{L}_{\mathrm{edit}}(a)
\Big),
\label{eq:repair_obj}
\end{equation}
where the feasible repair set is
\begin{equation}
\mathcal{A}_t^{\mathrm{feasible}}
=
\{a\in\mathcal{A}_t \mid S(G_{t+1}(a),X)\ge\tau_{\mathrm{pass}}\}.
\label{eq:repair_feasible}
\end{equation}
Here $\tau_{\mathrm{pass}}=0.5246$ is the data-driven repair trigger threshold (the 25th percentile of realized quality scores across training episodes), which is computed once from training data and held fixed at evaluation time.

This formulation separates repair from initial topology selection:
the initial stage chooses a globally feasible workflow under hard constraints, whereas repair performs \emph{local correction} aimed at restoring execution quality without global replanning.

\begin{algorithm}[t]
\caption{Hierarchical Pareto Orchestration}
\label{alg:hpo}
\begin{algorithmic}
\Require Task $X$, profile $\mathcal{P}$, budget $\Lambda$, deadline $\Delta$

\State $b \gets \textsc{EstimateDifficulty}(X)$
\State $T_0 \gets \textsc{SelectTemplate}(\mathcal{P}, X, b, \Lambda, \Delta)$
\For{$v \in V(T_0)$}
    \State $\phi(v) \gets \textsc{SelectExecutor}(\mathcal{P}, v, X, b, \Delta)$
\EndFor
\State Execute workflow $G_0=(T_0,\Phi_0)$
\If{any node fails}
    \State Generate repair candidates
    \State Select $a^* = \arg\max (Q - \lambda L_{\text{edit}})$
    \State Apply repair and update workflow
\EndIf
\State Update profiles and return workflow $G^*$
\end{algorithmic}
\end{algorithm}


\section{Experiments}
\label{sec:experiments}

We evaluate TopoGuard as a closed-loop topology orchestration framework. 
For each task instance, the system first selects an executable workflow under quality--cost--latency constraints and then monitors execution for possible local repair. 
The experiments are organized around the following questions:

\begin{itemize}
    \item \textbf{RQ1:} Can TopoGuard achieve a strong overall trade-off among quality, cost, and latency?
    \item \textbf{RQ2:} Does local repair contribute substantively to the final system performance?
    \item \textbf{RQ3:} Do the observed advantages remain stable across task domains and different utility preferences?
    \item \textbf{RQ4:} Does bounded local repair provide runtime robustness against unknown failures not reflected in profiles?
\end{itemize}

\subsection{Experimental Setup}
\label{subsec:exp_setup}



\paragraph{Platform and task domains.}
We conduct all experiments on a water-conservancy digital-twin platform and evaluate TopoGuard on two representative task domains: \textbf{Water QA} and \textbf{Storm Surge}. 
The primary benchmark, \textbf{Water QA}, is formulated as a multi-stage question answering task over hydrological knowledge. 
Different task instances may require different execution structures, such as retrieval, reasoning, computation, verification, and aggregation. 
The auxiliary benchmark, \textbf{Storm Surge}, is used to examine whether the learned orchestration strategy can transfer to another decision domain without redesigning the core decision logic.

\begin{figure}[t]
    \centering
    \includegraphics[width=0.88\textwidth]{outputs/overall_water_qa/figures/HW.png}
    \caption{The water-conservancy digital-twin agent platform used for evaluation. The platform integrates multimodal monitoring data, a task decomposition interface, and a pool of executable domain agents supporting heterogeneous workflow configurations.}
    \Description{Screenshot of the water-conservancy digital-twin agent platform showing task decomposition, agent pool, and monitoring interface.}
    \label{fig:platform}
\end{figure}

\paragraph{Data and evaluation protocol.}
For Water QA, we construct a benchmark containing training and test executions, which are further aggregated into historical performance profiles for candidate workflows and executors. 
For Storm Surge, we build an auxiliary transfer benchmark with heterogeneous execution choices and evaluate whether the same orchestration strategy can generalize across tasks. 
In both domains, the training split is used to estimate the expected quality, cost, and latency of candidate workflows, while all final results are reported on held-out test executions.

Given a test context, the system first queries historical profiles to estimate the expected triplet $(S,C,L)$ for each candidate topology or executor, applies hard constraint filtering, computes the feasible Pareto frontier, and then selects an execution plan according to the corresponding orchestration strategy.
For TopoGuard, the selected plan is further executed in a closed loop with evaluator monitoring and optional local repair.

Table~\ref{tab:benchmark_stats} summarizes the benchmark construction statistics. The Water QA benchmark comprises 17 models $\times$ 5 node types $\times$ 3 difficulty levels, yielding 163 non-empty profile entries (some combinations lack training coverage; notably, (retrieval, hard) has no valid candidates, affecting 17 test contexts). The Pareto frontier contains 46 feasible candidates, from which all strategies select. All profiles are estimated from training episodes on the same platform; no external data is used.

\begin{table}[t]
\centering
\caption{Benchmark construction statistics for the Water QA domain.}
\label{tab:benchmark_stats}
\small
\begin{tabular}{lr}
\toprule
\textbf{Item} & \textbf{Count / Value} \\
\midrule
Candidate models & 17 \\
Node types & 5 (retrieval, reasoning, computation, verification, aggregation) \\
Difficulty levels & 3 (easy, medium, hard) \\
Topology templates & 4 (direct, bad\_direct, ex+ver, ex+ver+agg) \\
Profile entries & 163 \\
Pareto frontier size & 46 \\
Training records & 1,467 \\
Test contexts (unique) & 255 \\
Test episodes (total) & 834 \\
Constraint budget $C_{\max}$ & 0.5 (log-normalized) \\
Latency budget $L_{\max}$ & 0.90 (log-normalized) \\
$Q$-score scale $S_{\text{scale}}$ & 1.5 \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Compared methods.}
We compare TopoGuard with six alternative orchestration strategies, organized by whether they select from the Pareto feasible candidate pool:

\textbf{Candidate-pool selection methods} (selecting from the 46 feasible candidates):
\begin{itemize}
    \item \textbf{TopoGuard (dual-layer adaptive):} performs Pareto pruning and utility-maximization selection at both the template level and the executor level, with bounded local repair. This is the full framework proposed in this paper.
    \item \textbf{LLM Router (executor-only adaptive):} selects the best executor by estimated quality within a fixed topology template (executor+verifier), without adapting workflow structure. Models the common model-routing approach and isolates the contribution of topology-level adaptation.
    \item \textbf{Random:} uniformly samples a candidate from the feasible Pareto frontier, serving as a no-adaptation baseline.
    \item \textbf{Best-Quality / Cheapest:} select the candidate with the highest estimated quality or the lowest estimated cost, serving as single-objective baselines.
\end{itemize}

\textbf{Fixed-pipeline methods} (not selecting from the candidate pool):
\begin{itemize}
    \item \textbf{Static Workflow:} fixes the topology to executor+verifier and the executor to kimi\_k2\_5. No Pareto screening, no adaptive topology selection, and no repair. This conservative non-adaptive pipeline serves as an alternative operating point for evaluating the quality-overhead trade-off.
    \item \textbf{FrugalGPT Cascade:} sorts executors by cost and selects the first meeting $S \geq 0.70$, without topology adaptation. Models the ``call cheap model first, escalate only if needed'' strategy.
\end{itemize}

This organization highlights the core innovation of TopoGuard---dual-layer adaptive topology selection---and provides controlled comparisons that isolate the contribution of each adaptive layer. The Static Workflow comparison evaluates whether the adaptive overhead is worthwhile in practice, rather than competing on a single metric.


\paragraph{Metrics.}
We report three primary metrics: \textbf{quality} $S\in[0,1]$, \textbf{cost} $C$ (USD), and \textbf{latency} $L$ (seconds).
Quality measures the final task-solving effectiveness of the selected workflow.
Cost is computed from actual execution expenses, and latency records the realized end-to-end execution delay.
Following the current implementation, cost and latency are log-normalized only for internal Pareto filtering, while all reported results use raw values for interpretability.

Note that the \textbf{violation rate (Viol\%)} is \emph{zero by construction} in our controlled selection experiment: all strategies select from a hard-filtered feasible set that excludes candidates violating the budget or latency constraints. This guarantees ex-ante feasibility for all methods, making Viol\% undifferentiated in this setting. Violation rate is meaningful only in a separate robustness or profile-drift experiment where the estimated profiles may mismatch actual execution conditions (see Exp~3 in the ablation analysis).

A related auxiliary metric is the \textbf{valid context rate}: the fraction of test contexts for which at least one feasible candidate exists in the profile (i.e., $N / N_{\text{total}}$ in the result tables). A context is \emph{invalid} if every candidate profile for that (node\_type, difficulty) pair violates either the budget constraint ($C > B$) or the latency constraint ($L > \Delta$). Context coverage directly reflects the feasibility of the candidate pool under given constraints; a higher valid context rate indicates broader operational coverage.

The valid context count $N$ varies across strategies. The primary source is missing profile coverage for the (retrieval, hard) pair (17 contexts affected), which causes Random ($N=239$) and Cheapest ($N=243$) to fall below 255; TopoGuard, Best-Quality, Static Workflow, and w/o Adaptive Template all cover the full 255 contexts. These N differences reflect strategy-specific feasibility coverage and are reported honestly in the result tables.

The internal selection metric $Q(G;X)=\alpha S_{\text{norm}}-\beta C_{\text{norm}}-\gamma L_{\text{norm}}$ is used solely for Pareto frontier ranking and candidate tie-breaking; it does not appear as a reported evaluation metric. $S_{\text{norm}}=S / S_{\text{scale}}$ normalizes quality to the same scale as $C_{\text{norm}}$ and $L_{\text{norm}}$, preventing quality from dominating the score purely due to scale differences. In our implementation, $(\alpha,\beta,\gamma)=(0.65,0.25,0.10)$ with $S_{\text{scale}}=1.5$, which balances quality prioritization with genuine cost-latency trade-offs on the Pareto frontier.

\subparagraph{Experiment protocol.}\label{subsubsec:exp_protocol}
Our experiments consist of two complementary evaluation protocols:
\textbf{(1) Shared-pool controlled comparison}: Random, Best-Quality, and Cheapest are evaluated as controlled selection baselines on the same learned profiles, selecting from the shared feasible candidate pool. This isolates the effect of the decision heuristic while holding the candidate space fixed.
\textbf{(2) Operating-point comparison}: TopoGuard and Static Workflow are evaluated on the same test contexts but represent different deployment philosophies (adaptive vs.\ fixed). They are not compared on a single metric; instead, the two are contrasted on their respective strengths: TopoGuard on quality and adaptive robustness, Static Workflow on cost-efficiency and simplicity.

\paragraph{Implementation details.}
The full TopoGuard pipeline consists of: 
(1) initial context analysis, 
(2) workflow-level candidate screening, 
(3) node-level executor selection, 
(4) execution and evaluator monitoring, and 
(5) bounded local repair. 
The repair operators are defined as three types: topology/template upgrade (operator A), executor upgrade (operator B), and evaluator upgrade (operator C). In the current closed-loop experiment, all triggered repairs fall under operator B (executor upgrade within ex+ver+agg topologies), as detailed in Section~\ref{subsec:ablation_repair}. Operator A and C are defined in the framework and available at runtime, but were not triggered in this evaluation set, consistent with the low overall failure rate (2.22\%) and the profile-based selection already providing strong initial executors.

\subsection{Main Results on Water QA}
\label{subsec:main_results}

\begin{figure}[t]
    \centering
    \includegraphics[width=0.88\textwidth]{outputs/overall_water_qa/figures/figA_strategy_comparison.png}
    \caption{Strategy comparison on Water QA (500 rounds). TopoGuard ($S=0.735$) achieves comparable quality to Best-Quality ($S=0.754$) at substantially lower cost ($C=0.477$ vs $0.582$) and latency ($L=121.9$\,s vs $141.4$\,s); Static Workflow achieves the lowest cost and latency ($C=0.001$, $L=43.4$\,s) as a fixed conservative pipeline. Paired comparison vs Static: 57.2\% win / 1.2\% tie / 41.6\% lose (n=255). TopoGuard also Pareto-dominates Static on both S and C in 29.0\% of contexts.}
    \Description{Three bar charts comparing TopoGuard, Static Workflow, Random, Best-Quality, and Cheapest on the Water QA benchmark across 500 test rounds. TopoGuard achieves the strongest quality-cost balance among adaptive methods.}
    \label{fig:strategy_comparison}
\end{figure}

We first examine \textbf{RQ1}, namely whether TopoGuard reaches a favorable trade-off among quality, cost, and latency on the Water QA benchmark.

Table~\ref{tab:water_qa_main} reports the closed-loop results on 500 held-out test rounds (834 episodes in total). Valid context counts (N) differ across strategies because profile coverage is incomplete for the (retrieval, hard) pair: TopoGuard and Static Workflow cover all 255 contexts, whereas Random ($N=239$) and Cheapest ($N=243$) cover fewer.
The comparison is best interpreted as a comparison between operating points rather than as a search for a single uniformly dominant method.

\textbf{TopoGuard (adaptive)} reaches $S=0.735$ through profile-guided topology selection with bounded repair. Relative to Best-Quality, it keeps quality close while lowering cost and latency ($C=0.477$ vs.\ $0.582$, $L=121.9$\,s vs.\ $141.4$\,s). On paired per-context comparison against Static Workflow ($n=255$), TopoGuard wins in 57.2\% of contexts, ties in 1.2\%, and loses in 41.6\%; it also Pareto-dominates Static on both $S$ and $C$ in 29.0\% of cases.

\textbf{Static Workflow (fixed)} represents a conservative baseline built from a predetermined executor+verifier template on a mid-tier model. It reaches lower quality ($S=0.703$), but at much lower cost ($C=0.001$) and latency ($L=43.4$\,s). The large cost ratio mainly reflects the token pricing of kimi\_k2\_5 in our pricing table; in absolute terms, both approaches remain below \$1 per query. In practice, the relevant comparison is therefore the quality gain obtained for additional operational overhead.

\textbf{FrugalGPT Cascade} and \textbf{LLM Router} represent established baselines from the literature. FrugalGPT Cascade ($S=0.682$, $C=0.301$, $L=79.7$\,s) escalates greedily through executors ordered by cost until a quality threshold is met, without any topology adaptation. LLM Router ($S=0.685$, $C=0.368$, $L=95.3$\,s) selects the best executor within a fixed topology template, adapting only the model choice. Both underperform TopoGuard by $\Delta S \approx +0.050$. The gap between LLM Router and TopoGuard isolates the contribution of topology-level adaptation: fixing the topology removes a meaningful fraction of TopoGuard's quality advantage even when the executor is selected optimally.

Figure~\ref{fig:strategy_comparison} shows the same pattern visually. Best-Quality reaches the largest $S$, Static Workflow minimizes overhead, and TopoGuard lies between them: it stays close to Best-Quality in $S$ while avoiding part of the associated cost and latency. This is consistent with the intended use of Pareto-guided selection, namely to choose workable operating points instead of optimizing a single metric in isolation.

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.96\textwidth]{outputs/overall_water_qa/figures/figB_paired_scatter.png}
    \Description{Left: per-context scatter plot of TopoGuard quality vs Static Workflow quality, colored by win/tie/lose. Right: histogram of per-context quality advantage delta S.}
    \caption{Per-context paired comparison of TopoGuard vs Static Workflow ($n=255$). \textbf{Left:} each point is one test context; points above the diagonal indicate TopoGuard wins. TopoGuard wins in 57.2\% of contexts, ties in 1.2\%, and loses in 41.6\%. \textbf{Right:} distribution of per-context quality advantage $\Delta S = S_\text{TopoGuard} - S_\text{Static}$; mean $\Delta S = +0.032$.}
    \label{fig:paired_scatter}
\end{figure*}

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.96\textwidth]{outputs/overall_water_qa/figures/figC_cdf_advantage.png}
    \Description{CDF of per-context quality advantage delta S for TopoGuard vs Static Workflow (left) and TopoGuard vs Best-Quality (right).}
    \caption{Cumulative distribution of per-context quality advantage $\Delta S$. \textbf{Left:} TopoGuard vs Static Workflow — the distribution is right-shifted (mean $+0.032$), with 57.2\% of contexts where TopoGuard wins. \textbf{Right:} TopoGuard vs Best-Quality — the distribution is slightly left-shifted (mean $-0.019$), with 73.3\% ties and 26.7\% losses; TopoGuard trades a small quality gap for substantially lower cost ($C=0.477$ vs $0.582$) and latency ($L=121.9$\,s vs $141.4$\,s).}
    \label{fig:cdf_advantage}
\end{figure*}

\begin{table}[t]
\centering
\caption{Closed-loop topology orchestration results on Water QA (500 rounds, 834 test episodes). All strategies select from the hard-filtered feasible set (Viol\% = 0\% by construction). FrugalGPT Cascade and LLM Router cover all 255 contexts; the variation in N for Random and Cheapest reflects missing profile coverage for the (retrieval, hard) pair.}
\label{tab:water_qa_main}
\resizebox{0.98\linewidth}{!}{
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{Quality $S$ $\uparrow$} & \textbf{Cost (USD) $\downarrow$} & \textbf{Latency (s) $\downarrow$} & \textbf{N} \\
\midrule
TopoGuard                                     & 0.735 & 0.477 & 121.9 & 255 \\
Static Workflow                               & 0.703 & \textbf{0.001} & \textbf{43.4} & 255 \\
FrugalGPT Cascade~\cite{chen2023frugalgpt}    & 0.682 & 0.301 & 79.7  & 255 \\
LLM Router~\cite{kim2024llm}                  & 0.685 & 0.368 & 95.3  & 255 \\
Random                                        & 0.686 & 0.334 & 88.5  & 239 \\
Best-Quality                                  & \textbf{0.754} & 0.582 & 141.4 & 255 \\
Cheapest                                      & 0.668 & 0.286 & 76.3  & 243 \\
\bottomrule
\end{tabular}}
\end{table}

% \begin{figure*}[t]
%     \centering
%     \includegraphics[width=0.98\textwidth]{samples/figures/fig_strategy_comparison.png}
%     \caption{Strategy comparison on Water QA. TopoGuard achieves a stronger overall balance across quality, cost, and latency than the alternative orchestration strategies.}
%     \label{fig:strategy_comparison}
% \end{figure*}




\subsection{Pareto Structure Analysis}
\label{subsec:pareto_analysis}

To further interpret the trade-off, we examine the per-context quality differences and their cumulative distributions.

Figure~\ref{fig:paired_scatter} shows the paired comparison between TopoGuard and Static Workflow at the context level. Points above the diagonal correspond to contexts in which TopoGuard achieves higher quality. The win rate is 57.2\%, with a mean advantage of $\Delta S = +0.032$. This pattern suggests that the gain is spread across a majority of contexts rather than being concentrated in only a few cases.

Figure~\ref{fig:cdf_advantage} reports the cumulative distribution of $\Delta S$ for two pairwise comparisons. Relative to Static Workflow, the distribution is shifted to the right, which is consistent with the positive mean advantage. Relative to Best-Quality, the distribution is slightly shifted to the left (mean $\Delta S = -0.019$), while 73.3\% of contexts are ties. Taken together, these plots indicate that TopoGuard usually remains close to the quality-maximizing strategy while reducing cost and latency.



\subsection{Auxiliary Transfer Evidence: Storm Surge}
\label{subsec:transfer}

We also report auxiliary evidence on cross-task transfer by applying the same orchestration logic and profiles, calibrated on Water QA, to the Storm Surge benchmark without retraining.
Because the sample size is small ($N \approx 20$ per strategy), we treat this analysis as supporting evidence rather than as a primary result.

Table~\ref{tab:storm_transfer} shows that TopoGuard reaches $S=0.853$ in the zero-shot transfer setting, higher than Static Workflow ($S=0.716$) and close to Best-Quality ($S=0.848$), with slightly lower cost.
An additional point of interest is the change in preferred topology: TopoGuard selects direct execution (50.0\%) and ex+ver (37.3\%) more often on Storm Surge, whereas ex+ver+agg is dominant on Water QA (71.1\%). Static Workflow remains fixed at ex+ver regardless of domain.
This shift suggests that the selection policy is sensitive to task structure rather than tied to one fixed template.

Given the small $N$, the transfer result should be read cautiously. The quality differences among adaptive methods are suggestive, but not strong enough to support a broad cross-domain claim on their own.

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.96\textwidth]{outputs/overall_water_qa/figures/figE_topo_heatmap.png}
    \Description{Heatmap of topology selection frequency for each strategy across Water QA and Storm Surge domains.}
    \caption{Topology preference heatmap across two domains. Each cell shows the selection frequency for a given topology. TopoGuard adapts its topology preference to the task domain (ex+ver+agg dominant in Water QA; direct and ex+ver preferred in Storm Surge), while Static Workflow is always fixed to ex+ver. This shift is the primary transfer signal: the selection mechanism responds to task structure rather than applying a domain-agnostic policy.}
    \label{fig:topo_heatmap}
\end{figure*}

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.96\textwidth]{outputs/overall_water_qa/figures/figI_cross_domain.png}
    \Description{Side-by-side comparison of TopoGuard and Static Workflow on Water QA and Storm Surge across quality, cost, and latency.}
    \caption{Cross-domain transfer (auxiliary). TopoGuard achieves higher quality than Static Workflow in both domains ($S=0.735$ vs $0.703$ on Water QA; $S=0.853$ vs $0.716$ on Storm Surge). Quality differences among adaptive strategies on Storm Surge ($N \approx 20$) are indicative rather than conclusive; the main observation is the consistent topology adaptation across domains.}
    \label{fig:cross_domain}
\end{figure*}

\begin{table}[t]
\centering
\caption{Auxiliary transfer results on Storm Surge (150 rounds, $N \approx 20$ per strategy). Results are indicative rather than conclusive due to the small sample. The primary transfer signal is the topology shift: TopoGuard selects lighter structures (direct 50.0\%, ex+ver 37.3\%) compared to Water QA (ex+ver+agg 71.1\%), while Static Workflow is always fixed to ex+ver.}
\label{tab:storm_transfer}
\resizebox{0.98\linewidth}{!}{
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{Quality $S$ $\uparrow$} & \textbf{Cost $\downarrow$} & \textbf{Latency (s) $\downarrow$} & \textbf{N} \\
\midrule
TopoGuard        & \textbf{0.853} & 0.046 & 16.7 & 20 \\
Static Workflow  & 0.716           & \textbf{0.024} & 40.8 & 23 \\
Random           & 0.837           & 0.045 & 16.1 & 16 \\
Best-Quality     & 0.848           & 0.047 & 15.5 & 20 \\
Cheapest         & 0.831           & 0.046 & 17.6 & 18 \\
\bottomrule
\end{tabular}}
\end{table}

\subsection{Component Ablation and Repair Analysis}
\label{subsec:ablation_repair}

This section addresses \textbf{RQ2}: how much each component contributes to final performance, and whether repair is useful when the initial selection underperforms.

\subparagraph{Core module ablation.}
Table~\ref{tab:ablation_repair} compares the full system with two ablated variants on the same 255 test contexts.
\textbf{w/o Adaptive Template} fixes the topology to executor+verifier and adapts only the executor selection. The resulting quality drops to $S=0.685$ ($\Delta S=-0.050$ relative to the full system). Latency is lower ($L=95.3$\,s) because the fixed structure is simpler, but the quality reduction indicates that template adaptation carries independent value.
\textbf{w/o Adaptive Executor} fixes the executor to qwen\_14b and adapts only the template. Coverage falls to $N=51$ (20\%), so its average quality ($S=0.742$) is not directly comparable to the full-coverage variants. The high mean largely reflects selection over a smaller subset of favorable contexts.
Overall, the ablation suggests that the observed gain comes from combining both adaptive layers rather than from either one alone.

\begin{table}[t]
\centering
\caption{Core module ablation on Water QA (500 rounds). Each variant removes one adaptive layer while holding the other fixed. w/o Adaptive Executor covers only 51/255 contexts (20\% coverage) due to the restricted executor choice; its quality is not directly comparable to full-coverage variants.}
\label{tab:ablation_repair}
\resizebox{0.98\linewidth}{!}{
\begin{tabular}{lcccc}
\toprule
\textbf{Variant} & \textbf{Quality $S$ $\uparrow$} & \textbf{Cost (USD) $\downarrow$} & \textbf{Latency (s) $\downarrow$} & \textbf{N} \\
\midrule
TopoGuard (full)              & \textbf{0.735} & 0.477 & 121.9 & 255 \\
\quad w/o Adaptive Template   & 0.685          & 0.368 & 95.3  & 255 \\
\quad w/o Adaptive Executor   & 0.742$^\dagger$& 0.358 & 108.5 & \phantom{0}51 \\
\bottomrule
\end{tabular}}
{\small $^\dagger$ Coverage 20\%; not directly comparable to full-coverage variants.}
\end{table}

\subparagraph{Repair mechanism analysis (RQ4).}
The repair module is intended for runtime failures that are not fully captured by the initial profile-based selection. In the 500-round closed-loop evaluation, 15 of 675 execution contexts trigger a repair (2.22\%), which indicates that repair is invoked infrequently rather than used as a routine part of every run.

All 15 triggered repairs are executor upgrades (operator B) within ex+ver+agg topologies. In these cases, qwen\_14b (6 cases), minimax\_ (6 cases), and qwen\_397 (3 cases) are replaced with stronger alternatives retrieved from the adaptive profile store. No topology-level (operator A) or evaluator-level (operator C) repairs are observed in this evaluation. This pattern is consistent with the main source of error in the current setting: executor mismatch rather than template failure or evaluator failure.

The per-trigger quality gain ranges from $\Delta S=+0.022$ to $+0.533$, with a mean of $+0.290$, and all 15 cases are resolved through adaptive profile lookup without falling back to preset estimates (Table~\ref{tab:repair_stats}). Averaged over all contexts, the contribution of repair is modest ($+0.290 \times 0.0222 \approx +0.006$), but it is consistently positive. In this sense, repair acts as a bounded safeguard for a small subset of difficult cases rather than as a high-frequency recovery mechanism.

\begin{table}[t]
\centering
\caption{Repair mechanism statistics over 675 execution contexts across 500 rounds. All triggered repairs are executor upgrades (operator B); topology and evaluator upgrades (operators A, C) are not triggered in this setting, consistent with the low overall failure rate and strong initial profile coverage.}
\label{tab:repair_stats}
\resizebox{0.98\linewidth}{!}{
\begin{tabular}{lcc}
\toprule
\textbf{Statistic} & \textbf{Value} & \textbf{Interpretation} \\
\midrule
Repair trigger rate         & 2.22\% (15/675)        & selective activation \\
Average quality gain        & $+0.290$ / trigger     & targeted recovery \\
Global quality contribution & $\approx +0.006$       & bounded overhead benefit \\
Repair operator type        & Executor upgrade (B)   & profile-guided replacement \\
Repair source               & 100\% adaptive lookup  & no preset fallback used \\
Repair targets              & qwen\_14b: 6, minimax\_: 6, qwen\_397: 3 & executor-level correction \\
\bottomrule
\end{tabular}}
\end{table}

\begin{figure*}[t]
  \centering
  \includegraphics[width=0.98\textwidth]{outputs/overall_water_qa/figures/figH_repair_impact.png}
  \caption{\textbf{Repair mechanism analysis (RQ4).} \textbf{Left:} repair trigger count per round across 500 test rounds. Total: 15 repairs out of 675 execution contexts (2.22\%). \textbf{Middle:} per-trigger quality gain $\Delta S$ (mean $+0.290$, range $+0.022$ to $+0.533$). \textbf{Right:} repair strategy distribution — all 15 cases are executor upgrades (operator B) in ex+ver+agg topologies. No topology or evaluator upgrades (operators A, C) are triggered in this evaluation.}
  \label{fig:repair_analysis}
\end{figure*}

\begin{figure}[t]
    \centering
    \includegraphics[width=0.8\textwidth]{outputs/overall_water_qa/figures/figF_waterfall.png}
    \Description{Waterfall chart showing cumulative quality contribution from Static Workflow baseline through each TopoGuard component to the full system.}
    \caption{Quality improvement waterfall. Starting from Static Workflow ($S=0.703$), adaptive template selection raises quality to $S=0.735$ (full TopoGuard, $\Delta S=+0.050$ vs.\ w/o Adaptive Template). Repair contributes a further bounded increment (global contribution $\approx +0.006$) on the 2.22\% of contexts where it triggers.}
    \label{fig:waterfall}
\end{figure}

Figures~\ref{fig:repair_analysis} and \ref{fig:waterfall} provide a more concrete view of this behavior.
Figure~\ref{fig:repair_analysis} shows that repair is selective and that the triggered cases receive positive quality corrections.
Figure~\ref{fig:waterfall} places this effect in the broader quality breakdown: most of the gain comes from adaptive topology selection, while repair adds a smaller improvement on the minority of contexts in which it is triggered.

\subsection{Sensitivity to Utility Weights}
\label{subsec:sensitivity}

\textit{Note: The sensitivity and profile-drift analyses in this section use an earlier 238-context protocol with the same candidate pool and profiles as the main experiment. The absolute $S$ values differ slightly from the 500-round evaluation because the context set is smaller, so these results should be read as supplementary evidence.}

We next consider \textbf{RQ3} from the perspective of the utility weights.
A useful orchestration policy should react to changes in the utility weights in a predictable way, without becoming unstable under moderate reweighting.

To test this behavior, we vary $(\alpha,\beta,\gamma)$ and re-score candidates on the same held-out contexts without retraining the profiles. 
Seven representative settings are considered, including the default configuration, quality-dominant weighting, balanced weighting, cost-priority weighting, latency-priority weighting, and two partial-objective variants.

Table~\ref{tab:sensitivity_simple} shows two main patterns.
First, the selected workflows shift in an interpretable way as the weights change. Under quality-dominant weighting ($\alpha=0.92$), the selected topologies move toward the deepest structures and the resulting quality reaches $S=0.735$; under latency-priority weighting ($\gamma=0.80$), the feasible set shrinks ($N=215$) and the selected workflows move toward faster, cheaper templates, with a lower quality of $S=0.616$.
Second, across the tested settings, the quality range remains bounded ($S\in[0.62,0.74]$). This suggests that the method is sensitive to utility preferences, but not excessively fragile to weight changes.


\begin{table}[t]
\centering
\caption{Sensitivity analysis under different utility weight settings (supplementary, earlier 238-context evaluation). The best quality value(s) are boldfaced. Our default setting ($\alpha=0.65,\beta=0.25,\gamma=0.10$, $S_{\text{scale}}=1.5$) balances quality prioritization with genuine cost-latency trade-offs on the Pareto frontier. The qualitative pattern --- predictable topology shifts under weight changes, stable quality range $S\in[0.62,0.74]$ --- is consistent with the full 500-round experiment.}
\label{tab:sensitivity_simple}
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Weight Setting} & $\alpha$ & $\beta$ & $\gamma$ & \textbf{$S$ $\uparrow$} \\
\midrule
Default (quality-centric) & 0.65 & 0.25 & 0.10 & 0.725 \\
Quality-dominant            & 0.92 & 0.05 & 0.03 & \textbf{0.735} \\
Balanced                   & 0.50 & 0.30 & 0.20 & 0.668 \\
Cost-priority               & 0.40 & 0.50 & 0.10 & 0.682 \\
Latency-priority           & 0.10 & 0.10 & 0.80 & 0.616 \\
Q+C only                    & 0.80 & 0.20 & 0.00 & \textbf{0.735} \\
Q+L only                    & 0.80 & 0.00 & 0.20 & 0.682 \\
\bottomrule
\end{tabular}
\end{table}


\subparagraph{Robustness to profile estimation error (supplementary).}
We also examine how the method behaves when the estimated profiles differ from realized execution conditions. This analysis again uses the earlier 238-context protocol. Profile drift is simulated by multiplying the estimated cost and latency used during topology selection by factors from $1.0\times$ to $1.5\times$, and then measuring realized performance on the held-out set.

Table~\ref{tab:drift_results} indicates that TopoGuard is relatively insensitive to cost estimation error: increasing the estimated cost by up to 50\% does not change quality ($S=0.728$), because cost mainly affects tie-breaking on the Pareto frontier. Latency error has a larger effect. When estimated latency is inflated by 50\%, selection shifts toward cheaper templates and quality decreases from $0.728$ to $0.692$. Even in this case, the degradation remains bounded and stays within the range observed in the weight-sensitivity analysis.

\begin{table}[t]
\centering
\caption{Robustness to profile estimation error --- supplementary analysis (earlier 238-context evaluation). Cost and latency multipliers are applied to the estimated values during topology selection. Reported $S$, $C$, $L$ are actual realized values on the held-out test set. Cost drift has negligible effect on quality; latency drift causes bounded quality degradation within the range observed in the sensitivity analysis.}
\label{tab:drift_results}
\small
\begin{tabular}{lccc}
\toprule
\textbf{Drift Scenario} & \textbf{$S$ $\uparrow$} & \textbf{$C$ $\downarrow$} & \textbf{$L$ $\downarrow$} \\
\midrule
No drift         & 0.725 & 0.460 & 116.3 \\
Cost +20\%       & 0.725 & 0.460 & 116.3 \\
Cost +50\%       & 0.725 & 0.460 & 116.3 \\
Latency +20\%    & 0.722 & 0.396 & 104.1 \\
Latency +50\%    & 0.692 & 0.301 & 84.6  \\
Both +25\%       & 0.716 & 0.374 & 101.1 \\
Both +50\%       & 0.692 & 0.301 & 84.6  \\
\bottomrule
\end{tabular}
\end{table}



The aggregate results characterize the overall trade-off, but they do not show how the framework behaves within a single run. To make the runtime behavior more concrete, we include a case study based on a storm-surge warning task, focusing on what happens when a downstream node becomes mismatched with the intermediate outputs it receives.

\subsection{Case Study: Execution and Local Repair in a Storm-Surge Scenario}

We conduct the case study on a water-conservancy digital-twin agent platform. The platform integrates three key components: multimodal monitoring data, a task decomposition interface, and a pool of executable domain agents. This environment provides a realistic setting for evaluating how TopoGuard organizes available executors into an executable workflow for storm-surge warning tasks.

Given a user request and the available tool set, TopoGuard first generates an initial execution topology and assigns a default downstream analysis node. The forecast module is then invoked to produce intermediate outputs for subsequent interpretation. In our case, these outputs are not simple scalar values, but a collection of heterogeneous artifacts, including temporal evolution curves, spatial comparison results at a key time step, metric summaries, and spatial error maps.

These forecast outputs are passed to the downstream analysis node as intermediate evidence. However, the default analysis node fails to adequately interpret this heterogeneous output package and therefore cannot produce a satisfactory downstream result. Instead of replanning the entire workflow, TopoGuard triggers bounded local repair and replaces only the current analysis node, while preserving the forecast outputs generated upstream.

\begin{figure}[t]
    \centering
    \includegraphics[width=0.88\textwidth]{outputs/overall_water_qa/figures/forecast_outputs.png}
    \caption{Heterogeneous forecast outputs produced by the storm-surge forecast module in the case study. The outputs include temporal evolution curves, spatial comparison results at a key time step, metric summaries, and spatial error maps — representing the multimodal intermediate evidence that the downstream analysis node must interpret.}
    \Description{Forecast module outputs for a storm-surge task instance, showing temporal curves, spatial maps, and metric summaries as intermediate evidence passed to the downstream analysis node.}
    \label{fig:forecast_outputs}
\end{figure}

After repair, the system resumes execution from the same forecast evidence and continues the warning-generation process. The case illustrates two practical aspects of the framework: it can construct an initial executable topology for a realistic multimodal platform, and it can correct a local mismatch without discarding previously computed upstream outputs.


\subsection{Discussion}
\label{subsec:discussion}

The results are most naturally interpreted as evidence about operating points under constrained orchestration. TopoGuard improves quality relative to a fixed static workflow ($S=0.735$ vs.\ $0.703$), but it does so with higher cost and latency. At the same time, it remains cheaper and faster than the quality-maximizing baseline while keeping quality close to it. In that sense, the main result is not that TopoGuard dominates every alternative on every metric, but that it occupies a useful part of the feasible quality--cost--latency space under the offline-profiled evaluation protocol.

The robustness claim should also be read with the structure of the experiments in mind. The main selection experiment has Viol\% equal to zero by construction, so the more direct evidence for runtime recovery comes from the repair and profile-drift analyses. In the current evaluation, repairs are infrequent (15 of 675 contexts) and all of them are executor upgrades, which suggests that the dominant failure mode in this setting is local executor mismatch rather than large topology failure. This is a narrower claim than saying that all forms of runtime failure are solved, but it is still useful: bounded local repair improves a small but important subset of difficult cases without adding overhead to most runs.

\section{Conclusion}
\label{sec:conclusion}

This paper studies topology orchestration for multi-stage decision tasks under multimodal uncertainty and hard quality--cost--latency constraints. The proposed framework, TopoGuard, combines profile-based topology selection, node-level executor assignment, and bounded local repair in a single closed-loop procedure. Experiments on a water-conservancy digital-twin platform show that this design yields a better quality operating point than a fixed static workflow, while staying cheaper and faster than the quality-maximizing alternative at similar quality. Supplementary analyses further suggest that the method responds predictably to utility-weight changes and degrades in a bounded way under moderate profile mismatch. Taken together, these results support topology-level adaptation as a practical design choice for constrained multimodal decision systems.


\clearpage
\newpage
\bibliographystyle{ACM-Reference-Format}
\bibliography{reference}

\end{document}
