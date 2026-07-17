

## interfaces

- Do NOT proliferate command line arguments. There should be ONE way to 
  achieve a result. The default should be the most user-friendly thing. If 
  you cannot answer the question "do I absolutely need this for a known or 
  anticipated use case", dont add extra options.

- For any long-running scripts (estimated to take more than 10s) please include 
  human-friendly phasing lines in the output along with a progbar that 
  shows an ETA for the phase and overall task.

- If any sequential, log-like data is generated and used in a report, it 
  should be written to disk in a temporary directory as generated 
  (batches are acceptable to avoid slowing down a loop) instead of 
  constructed in memory and dumped out only at the end. The script should report 
  the temporary artifact directory to stdout.


## typing

- Use proper type names. Eschew use of Any, and especially "Any | None" 



## functional programming

- Write code in a functional style by default with persistent data 
  structures (prefer pyrsistent library). 
  Do not introduce mutable state that breaks thread safety unless it is 
  critical to performance (and not just a constant overhead factor either).  
  Make a good argument and ask the user to approve before doing so if it 
  seems  to be needed. It probably isnt.

- Use comprehensions instead of for loops unless their bodies are large and 
  ungainly

## DRY

Dont write repetitive code. Dont repeat similar function calls with minor 
differences in arguments if they have more than 2 or 3 arguments.
Dont create separate codepaths when common core routines can be called with 
expanded arguments or easy-to-read branches in the arguments. 

```
# BAD
result1 = function_call_with_lots_of_args(a=derp, b=herp, ...., m=zerp)
result2 = function_call_with_lots_of_args(a=derp, b=herp, ...., 
                                          m=slightly_different_zerp, ...)

# BETTER
result1, result2 = [function_call_with_lots_of_args(a=derp, b=herp, ...,
        m=z, ... ) for z in [zerp, slightly_different_zerp]]  
```

In general, before returning from your task, ask if the code you wrote could 
be abstracted or condensed and made more concise while retaining functional 
style and clear semantics. If the cause of the unnecessary redundancy is 
outside the scope of what you just did, its okay to refactor to pull out 
common sub-functions, etc so that the rest of the code is in better shape. 

Don't create separate top level function definitions for single-use, 
expressions that could be inline, either at a call site or nearby with a 
concise lambda. Rule of thumb: if a function is only called once and its 
inline expression is shorter than the full body with type declarations and 
everything, prefer local lambda or call-site inlining.

## readability

Code can be read almost like natural language if it is structured well and 
avoids overly terse identifiers. Prefer longer, multi_word_identifiers for 
one-off variables; guts of a routine type stuff that is hard to read.

Short or abbreviated variable names are fine if they are extremely common and 
ubiquitously, and have unambigous meaning in context.

Limit complex (in the sense of behavior, not strictly line count) functions. 
Break them up in to sub functions with readable names. Multiple closures, 
reductions, loops, recursion, etc should be a cue to maybe refactor.



## separation of concerns

- Keep modules focused. Dont group utility, debugging, logging, or other 
  things within core algorithmic modules.
- Logging, tracing, diagnostics should have strictly minimal footprint at 
  call sites within core logic pathways. 


## comments and docstrings

- Document every public function that is more than 3 lines long with a 
  one-sentence docstring. 
- Document key or very complex functions with full docstrings including 
  examples.

## test styles
Not all "should" instructions that imply change from an previous 
implementation merit extensive test cases. For example, "such-and-such 
should take dependency injection instead of hard-coded registry for blah" 
DOES NOT NEED A BUNCH OF TEST CASES TO ENFORCE IT. Its sufficient to just 
refactor the constructors and call sites, as long as SOME test covers the 
new behavior (which they will, if its a pure refactor)