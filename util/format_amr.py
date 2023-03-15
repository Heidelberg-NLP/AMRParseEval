import re

def format_amr(amrstring):

    # clean wiki
    string = amrstring.strip()
    string = re.sub(r":wiki \"[^\"]+\"", "    ", string)
    string = re.sub(r":wiki \'[^\']+\'", "    ", string)
    string = re.sub(r":wiki -", "    ", string)
    string = string.replace("\n", "    ")
    
    # Rename variables

    # 1. collect all variables
    varr = re.findall(r"\([^ ]+ /", string)
    var = [v.replace(" /", "").replace("(", "") for v in varr]
    
    # 2. assign standardized index
    vm = {}
    idx = 0
    for v in var:
        vm[v] = "vx" + str(idx)
        idx += 1

    # 3. replace variables in string with new standardized index
    oldstring = string
    for v in vm:    
        
        # Replace newly introduced variable with standardized index
        # E.g., "(r / run-01" --> "(xvn / run-01"
        string = string.replace("("+v+" / ", "("+vm[v]+" / ")

        # Replace other mentions/re-entrancies of the variable with standardized index
        # E.g., ":ARG1 r "--> ":ARG1 xvn"
        string = string.replace(" "+v+"  ", " "+vm[v]+ "  ")
        
        # -------------------------------------------------------------#
        # Take care of varibale name = concept name issue 
        # and take care of re-entrant variables that conclude a subgraph
        # E.g. ... ":ARG1 (i / i)" ---> ":ARG1 (vxn / i)"
        # E.g. ... ":ARG1 i)"      ---> ":ARG1 xvn)"
        # -------------------------------------------------------------#
        
        # 3.1. mark a concept if it is also a variable
        # E.g., " / i)" ---> "/i)"
        string = string.replace("/ "+v+")", "/"+v+")")

        # 3.2. now we can replace re-entranties that conclude sub-graph
        # E.g., " i)" ---> " xvn)"
        string = string.replace(" "+v+")", " "+vm[v]+")")
        
        # 3.3. and then restore marked concepts
        # E.g., "/i)" ---> "/ i)" 
        string = string.replace("/"+v+")", "/ "+v+")")
    
    # 4. standardize white spaces 
    string = " ".join(string.split())
    return string

