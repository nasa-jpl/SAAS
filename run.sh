#! /bin/bash
sim-acs /home/j/Code/sync/saas/acs-syssim/configs/default.toml 
column -s, -t < /tmp/monsid_record.csv | less -#2 -N -S