#!/bin/bash

set -e

SPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INDRI=$SPATH/../../../../external/local/bin/IndriRunQuery

$INDRI \
    -baseline=okapi,k1:0.9,b:0.4,k3:0 \
    -index=gov2_indri \
    -stemmer.name=krovetz \
    -count=1000 \
    -trecFormat=1 \
    gov2-bow.qry > gov2-bow.run
