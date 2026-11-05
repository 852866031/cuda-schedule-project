# Greedy-output check (8 fixed natural-language prompts, 64 tokens, temperature 0, one request at a time)

| engine | prompts identical to plain | identical to kvcached+LMCache |
|---|---|---|
| plain (no connector) | 8/8 | 4/8 |
| kvcached, no connector | 8/8 | 4/8 |
| plain + LMCache MP engine_driven | 4/8 | 8/8 |
| kvcached + LMCache MP engine_driven | 4/8 | 8/8 |

Prompts that differ (plain vs. with the LMCache connector), first 90 characters:

- c0: plain `' the reign of Augustus Caesar (27 B.C.E. - 14 C.E.), who overthrew the Roman Republic and '`
  with connector `' the overthrow of the Roman Republic in 49 BC and ends with the fall of the Western Roman '`
- c1: plain `' a data structure that implements an associative array abstract data type, a structure tha'`
  with connector `' a data structure that implements an associative array abstract data type, a structure tha'`
- c2: plain `' is one of the most important scientific theories of the 20th century. It is a theory of g'`
  with connector `' is one of the most important scientific theories of the 20th century. It is a theory of g'`
- c4: plain `' its many iconic landmarks, including the Eiffel Tower, the Louvre, and the Arc de Triomph'`
  with connector `' its many beautiful parks and gardens. The most famous of these is the Jardin des Tuilerie'`

The divergence follows the LMCache connector, not kvcached: the plain engine with the connector reproduces the
kvcached+connector outputs exactly, and kvcached without the connector reproduces plain exactly. The runs were
repeated three times on the kvcached+connector engine with identical results.
