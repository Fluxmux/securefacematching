@echo off
SET ps_title="MPCServer"
FOR /L %%d IN (0 1 2) DO  (
   START "%ps_title%%%d" powershell docker-compose -p "server%%d" -f "docker-compose_%%d.yml" up --build --remove-orphans
)
EXIT