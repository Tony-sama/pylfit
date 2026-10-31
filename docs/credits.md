# Credits

pylfit's code is written and maintained by Tony Ribeiro, with the AI
assistants named at the end of this page. The theory it implements,
the Learning From Interpretation Transitions (LFIT) framework, was
developed by many people over more than ten years, and users shaped
what pylfit does. This page credits them: first the papers that
propose or extend an LFIT method, then the users who shaped pylfit,
then the work that applies LFIT, and last, how pylfit 1.0 was
written.

## LFIT methods

### Core team

- **Tony Ribeiro** co-authored every method paper below but those of
  other teams, listed last.
- **Katsumi Inoue** co-authored every method paper below, including the
  framework itself: "Learning from interpretation transition" (Machine
  Learning, 2014).
- **Chiaki Sakama** co-authored the framework paper (Machine Learning,
  2014), the BDD-based algorithm (ILP 2013), and the work on delayed
  influences: "Learning delayed influences of biological systems"
  (Frontiers, 2015) and "Learning multi-valued biological models with
  delayed influence from time-series observations" (ILP 2015,
  ICMLA 2015).
- **Morgan Magnin** co-authored the work on delayed influences (ILP
  2014, Frontiers 2015, ILP 2015, ICMLA 2015), continuous domains (ILP
  2017), `GULA` (ILP 2018; Machine Learning, 2022), `PRIDE` (IJCLR 2021,
  late-breaking abstract), the book chapter (2022, 2023), and the
  extensions to event diagnosis (2022), unknown values (2024) and
  counterfactual explanations (2025, 2026).
- **Maxime Folschette** co-authored continuous domains (ILP 2017),
  `GULA` (ILP 2018; Machine Learning, 2022), `PRIDE` (IJCLR 2021,
  late-breaking abstract), the book chapter "The challenges of inferring
  dynamic models from time series" (2022 in French, 2023 in English),
  and the extensions to event diagnosis (2022), unknown values (2024)
  and counterfactual explanations (2025, 2026).
- **Olivier Roux** co-authored continuous domains (ILP 2017), `GULA`
  (ILP 2018) and the book chapter (2022, 2023).

### Collaborators

- **Sophie Tourret**, **Domenico Borzacchiello** and **Francisco
  Chinesta** co-authored "Inductive learning from state transitions
  over continuous domains" (ILP 2017).
- **Laurent Trilling** and **Nicolas Glade** co-authored the book
  chapter "The challenges of inferring dynamic models from time series"
  (2022, 2023), which discusses `PRIDE`.
- **Yin Jun Phua** co-authored the work on learning delayed systems
  with neural networks: "Learning logic program representation for
  delayed systems with limited training data" (ILP 2017, with Sophie
  Tourret) and "Learning representation of relational dynamics with
  delays and refining with prior knowledge" (NeSy 2018; IfCoLog
  Journal of Logics and their Applications, 2019).
- **David Martínez**, **Guillem Alenyà** and **Carme Torras**
  co-authored "Learning probabilistic action models from
  interpretation transitions" (ICLP 2015).
- **Antoine Roquilly** and **Jérémie Poschmann** co-authored the work
  on unknowns (2024) and counterfactual explanations (2025, 2026).
- **Tuan Nguyen** co-authored "Counterfactual explanations under
  learning from interpretation transitions" (2025) and its follow-up
  (2026).
- **Nicolas Schwind** co-authored "Rule-based counterfactual
  explanations for learning from interpretation transitions" (2026).

### Collaborators outside academia

LFIT has also been developed with partners outside academia.

- **Kotaro Okazaki**, founder and former CEO of SONAR Inc. (Japan) and
  now Representative Director of Steelous Protocol, and **Kuo-Yen Lo**,
  Solution Architect at Steelous Protocol, co-authored "Diagnosis of
  event sequences with LFIT" (2022), "Learning from interpretation
  transition with unknowns" (2024) and the counterfactual-explanation
  papers (2025, 2026).

### LFIT methods by other teams

- **Yin Jun Phua** and **Katsumi Inoue** continued the neural-network
  line: "Learning logic programs from noisy state transition data"
  (ILP 2019), "Learning logic programs using neural networks by
  exploiting symbolic invariance" (ILP 2021) and "Variable assignment
  invariant neural networks for learning logic programs" (NeSy 2024).
- **Kun Gao**, **Hanpin Wang**, **Yongzhi Cao** and **Katsumi Inoue**:
  "Learning from interpretation transition using differentiable logic
  programming semantics" (Machine Learning, 2021).

## Users who shaped pylfit

- **Omar Ikne** asked for the rules behind each prediction
  ([issue #2](https://github.com/Tony-sama/pylfit/issues/2)), which
  pylfit 0.2.0's `DMVLP.predict` then returned; in 1.0, filtering
  `model.rules` with `Rule.matches` gives them. 0.x's
  `PRIDE.find_one_optimal_rule_of`, which learns a rule covering one
  given state, was written for his work, and his preliminary work on
  generalizing the rules `GULA` and `PRIDE` learn (IJCLR 2021) is what
  `generalize` does in 1.0.
- **Madeleine Eyraud** corrected a comment repeated in four of the 0.x
  examples
  ([pull request #4](https://github.com/Tony-sama/pylfit/pull/4)).
- **Juliette Audemard** used pylfit 1.0 during its development. Her
  feedback led to the `Dataset.positives()` and `Dataset.negatives()`
  accessors, with their results cached. Her prototype, on her fork,
  restricting the rule bodies of `GULA` and `PRIDE` with background
  knowledge led to `GULA`'s `allowed_conditions`.

## LFIT applications

Work that uses LFIT, rather than extending it:

- Explainable AI, by Alfonso Ortega, Julian Fierrez, Aythami Morales,
  Zilong Wang and Tony Ribeiro: `PRIDE` builds a rule-based twin of a
  neural-network recruitment classifier to explain it ("Symbolic AI for
  XAI: evaluating LFIT inductive programming for fair and explainable
  automatic recruitment", WACV 2021 workshops), followed by work on
  explaining biases with Marina de la Cruz and César Luis Alonso
  (Computers, 2021) and with Javier Tello and Ruben Tolosana (AEQUITAS
  2023 workshop).
- Planning and reinforcement learning, by David Martínez, Guillem
  Alenyà, Carme Torras, Tony Ribeiro and Katsumi Inoue: "Learning
  relational dynamics of stochastic domains for planning" (ICAPS 2016)
  and "Relational reinforcement learning for planning with exogenous
  effects" (JMLR, 2017).
- Recommendation systems, by Kotaro Okazaki, Tony Ribeiro and Kuo-Yen
  Lo (SONAR Inc.), Jyunichi Sakuma (Video Research Ltd.) and Katsumi
  Inoue: LFIT explains neural session-based recommenders on real
  viewing logs ("Data-driven analysis of domain specificity for
  explainable session-based recommendation system", JSAI 2024; code at
  [Tony-sama/JSAI-2024](https://github.com/Tony-sama/JSAI-2024)).
- Automotive fault diagnosis, by Shuichi Sato and Yoshinao Ishii
  (Toyota Central R&D Labs) with Yosuke Watanabe, Hiroyuki Seki and
  Shoji Yuen (Nagoya University): rules learned with ACEDIA, the LFIT
  algorithm for continuous domains, diagnose a cooperative automotive
  control system ("Fault diagnosis for distributed cooperative system
  using inductive logic programming", IEEE ICPHM 2020), a team with
  no LFIT author.
- Learning inference rules, by Chiaki Sakama, Katsumi Inoue and Tony
  Ribeiro: "Learning deduction rules by induction" (ILP 2015) and
  "Learning inference rules from data" (KI, 2019).
- Marine ecosystems, by Omar Ikne, Maxime Folschette and Tony Ribeiro:
  "Automatic modeling of dynamical interactions within marine
  ecosystems" (IJCLR 2021, late-breaking abstract).
- Marine ecology, by Madeleine Eyraud, Maxime Folschette, Katsumi
  Inoue, Sébastien Lefebvre and Cédric Lhoussaine: interaction graphs
  of phytoplankton species learned with LFIT ("Interaction graphs of
  phytoplankton species interactions using logical learning", CMSB
  2025). Madeleine Eyraud's PhD thesis, under way at the Université de
  Lille since 2023, is on learning interaction networks in marine
  ecosystems.

## How pylfit 1.0 was written

Tony Ribeiro wrote pylfit 1.0 with AI assistants, in three stages:

- May 2026: Yumi, a virtual assistant hosted locally on an open-weight
  large language model, scaffolded the repository and helped plan the
  rewrite.
- July 2026: Claude Sonnet 5, through the claude.ai web chat.
- From late July 2026: Claude Code, Anthropic's coding assistant, with
  Claude Sonnet 5, Claude Opus 5 and Claude Opus 5.5.

The assistants wrote most of the code, tests and documentation, from
Tony Ribeiro's design and instructions. He made the design decisions
and reviewed every change, and every change passes the tests and
checks described in [Contributing](contributing/index.md).
[`AGENTS.md`](https://github.com/Tony-sama/pylfit/blob/next/AGENTS.md)
holds the instructions the assistants work from. pylfit 0.x
(2019–2025, up to 0.5.1) was written without AI assistants.

If you know of work that proposes, extends or uses LFIT and is missing
here, open an issue.
