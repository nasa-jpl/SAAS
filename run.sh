#! /bin/bash
sim-acs /home/j/Code/sync/saas/acs-syssim/configs/default.toml -f /home/j/Code/sync/saas/acs-syssim/configs/basic_fault_config.toml
column -s, -t < /tmp/monsid_record.csv | less -#2 -N -S