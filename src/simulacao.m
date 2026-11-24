function simulacao()
% =========================================================================
% PME 3361 - Transferencia de Calor (Projeto I)
% Avaliacao do processo de colagem de pelicula plastica por laser em chapa
% de aco. Versao compativel com MATLAB e GNU Octave.
% Grupo 15C: h = 150 W/(m^2.K), q'' = 80000 W/m^2, T_inf = 25 C.
% =========================================================================

close all; clc;

% Parametros fixos do problema
p_fixos.rho      = 7850.0;    % Massa especifica do aco [kg/m^3]
p_fixos.cp       = 435.0;     % Calor especifico do aco [J/(kg.K)]
p_fixos.k        = 60.0;      % Condutividade termica do aco [W/(m.K)]
p_fixos.d        = 1.25e-3;   % Espessura da chapa [m]
p_fixos.w1       = 44.0e-3;   % Largura da pelicula [m]
p_fixos.w2       = 132.0e-3;  % Largura total da chapa [m]
p_fixos.L        = p_fixos.w2 / 2.0;
p_fixos.alpha    = p_fixos.k / (p_fixos.rho * p_fixos.cp);
p_fixos.dt_laser = 10.0;      % Duracao do pulso [s]
p_fixos.t_final  = 30.0;      % Tempo total de simulacao [s]

% -------------------------------------------------------------------------
% 1. Validacao obrigatoria (Benchmark Grupo 11A)
% -------------------------------------------------------------------------
res_11A = resolver_placa(100.0, 75000.0, 25.0, 200, 3000, p_fixos.w1, NaN, p_fixos);
met_11A = calcular_metricas(res_11A, p_fixos);

fprintf('========================================================================\n');
fprintf('VALIDACAO BENCHMARK - GRUPO 11A (N=200, Nt=3000)\n');
fprintf('========================================================================\n');
fprintf('Bi_y             = %.6e\n', res_11A.Bi_y);
fprintf('Bi               = %.6f   (esperado: 5.808000)\n', res_11A.Bi);
fprintf('q_tilde          = %.6f (esperado: 174.240000)\n', res_11A.q_tilde);
fprintf('T_max na pelicula= %.4f C (esperado: ~152.2 C)\n', met_11A.T_max_pulso);
fprintf('t_90 na borda    = %.4f s (esperado: ~9.30 s)\n', met_11A.t_90_borda);
fprintf('T_borda (t=10 s) = %.4f C (esperado: ~93.8 C)\n', met_11A.T_borda_10s);

assert(abs(res_11A.Bi - 5.808) < 1e-6, 'Bi divergente no benchmark 11A');
assert(abs(res_11A.q_tilde - 174.240) < 1e-6, 'q_tilde divergente no benchmark 11A');
assert(abs(met_11A.T_max_pulso - 152.2) < 0.3, 'T_max divergente no benchmark 11A');
assert(abs(met_11A.t_90_borda - 9.30) < 0.08, 't_90_borda divergente no benchmark 11A');
assert(abs(met_11A.T_borda_10s - 93.8) < 0.3, 'T_borda_10s divergente no benchmark 11A');
fprintf('STATUS BENCHMARK 11A: APROVADO COM SUCESSO!\n');
fprintf('========================================================================\n\n');

% -------------------------------------------------------------------------
% 2. Caso Base - Grupo 15C
% -------------------------------------------------------------------------
h_15C    = 150.0;
q_15C    = 80000.0;
Tinf_15C = 25.0;

res_15C = resolver_placa(h_15C, q_15C, Tinf_15C, 200, 3000, p_fixos.w1, NaN, p_fixos);
met_15C = calcular_metricas(res_15C, p_fixos);

fprintf('========================================================================\n');
fprintf('RESULTADOS DO CASO BASE - GRUPO 15C (N=200, Nt=3000)\n');
fprintf('========================================================================\n');
fprintf('alpha               = %.6e m^2/s\n', p_fixos.alpha);
fprintf('Bi_y                = %.6e\n', res_15C.Bi_y);
fprintf('Bi                  = %.6f\n', res_15C.Bi);
fprintf('q_tilde             = %.6f\n', res_15C.q_tilde);
fprintf('t_tilde_L (10 s)    = %.6f\n', res_15C.t_tilde_L);
fprintf('t_tilde_final (30s) = %.6f\n', res_15C.t_tilde_final);
fprintf('T_max (pulso)       = %.4f C\n', met_15C.T_max_pulso);
fprintf('T(0, 10 s)          = %.4f C\n', met_15C.T_centro_10s);
fprintf('T(11 mm, 10 s)      = %.4f C\n', met_15C.T_11mm_10s);
fprintf('T(22 mm, 10 s)      = %.4f C\n', met_15C.T_borda_10s);
fprintf('T(44 mm, 10 s)      = %.4f C\n', met_15C.T_44mm_10s);
fprintf('t_90_centro         = %.4f s\n', met_15C.t_90_centro);
fprintf('t_90_borda          = %.4f s\n', met_15C.t_90_borda);
fprintf('Janela [90,200] (p) = %.4f s\n', met_15C.tempo_janela_pulso);
fprintf('========================================================================\n\n');

% -------------------------------------------------------------------------
% 3. Estudo de Convergencia de Malha - Grupo 15C
% -------------------------------------------------------------------------
fprintf('ESTUDO DE CONVERGENCIA DE MALHA - GRUPO 15C\n');
malhas_N = [100, 200, 400];
Tc_N = zeros(size(malhas_N));
Tb_N = zeros(size(malhas_N));
for idx = 1:length(malhas_N)
    r_tmp = resolver_placa(h_15C, q_15C, Tinf_15C, malhas_N(idx), 3000, p_fixos.w1, NaN, p_fixos);
    m_tmp = calcular_metricas(r_tmp, p_fixos);
    Tc_N(idx) = m_tmp.T_centro_10s;
    Tb_N(idx) = m_tmp.T_borda_10s;
end
for idx = 1:length(malhas_N)
    ec = abs(Tc_N(idx) - Tc_N(end)) / Tc_N(end) * 100.0;
    eb = abs(Tb_N(idx) - Tb_N(end)) / Tb_N(end) * 100.0;
    fprintf('  N=%3d | T(0,10s)=%.4f C (err=%.4f%%) | T(22mm,10s)=%.4f C (err=%.4f%%)\n', ...
            malhas_N(idx), Tc_N(idx), ec, Tb_N(idx), eb);
end

passos_Nt = [1500, 3000, 6000];
Tc_Nt = zeros(size(passos_Nt));
Tb_Nt = zeros(size(passos_Nt));
for idx = 1:length(passos_Nt)
    r_tmp = resolver_placa(h_15C, q_15C, Tinf_15C, 200, passos_Nt(idx), p_fixos.w1, NaN, p_fixos);
    m_tmp = calcular_metricas(r_tmp, p_fixos);
    Tc_Nt(idx) = m_tmp.T_centro_10s;
    Tb_Nt(idx) = m_tmp.T_borda_10s;
end
for idx = 1:length(passos_Nt)
    ec = abs(Tc_Nt(idx) - Tc_Nt(end)) / Tc_Nt(end) * 100.0;
    eb = abs(Tb_Nt(idx) - Tb_Nt(end)) / Tb_Nt(end) * 100.0;
    fprintf('  Nt=%4d | T(0,10s)=%.4f C (err=%.4f%%) | T(22mm,10s)=%.4f C (err=%.4f%%)\n', ...
            passos_Nt(idx), Tc_Nt(idx), ec, Tb_Nt(idx), eb);
end
fprintf('\n');

% -------------------------------------------------------------------------
% 4. Solucoes de Engenharia Propostas
% -------------------------------------------------------------------------
res_A = resolver_placa(h_15C, 105000.0, Tinf_15C, 200, 3000, p_fixos.w1, NaN, p_fixos);
met_A = calcular_metricas(res_A, p_fixos);

res_B = resolver_placa(h_15C, q_15C, Tinf_15C, 200, 3000, 60.0e-3, NaN, p_fixos);
met_B = calcular_metricas(res_B, p_fixos);

q_2estagios = @(t_s) (t_s <= 2.0 + 1e-12) * 180000.0 + ...
                     (t_s > 2.0 + 1e-12 & t_s <= 10.0 + 1e-12) * 75000.0;
res_C = resolver_placa(h_15C, q_2estagios, Tinf_15C, 200, 3000, 60.0e-3, NaN, p_fixos);
met_C = calcular_metricas(res_C, p_fixos);

res_D = resolver_placa(h_15C, q_15C, Tinf_15C, 200, 3000, p_fixos.w1, 55.0, p_fixos);
met_D = calcular_metricas(res_D, p_fixos);

res_h100 = resolver_placa(100.0, q_15C, Tinf_15C, 200, 3000, p_fixos.w1, NaN, p_fixos);
met_h100 = calcular_metricas(res_h100, p_fixos);

dT_max_por_W   = (met_15C.T_max_pulso - Tinf_15C) / q_15C;
dT_borda_por_W = (met_15C.T_borda_10s - Tinf_15C) / q_15C;
q_min_90       = (90.0 - Tinf_15C) / dT_borda_por_W;
q_max_200      = (200.0 - Tinf_15C) / dT_max_por_W;

fprintf('SOLUCOES DE ENGENHARIA - GRUPO 15C\n');
fprintf('  Limites Sol. A (w=44mm): q_min(90C)=%.1f W/m^2 | q_max(200C)=%.1f W/m^2\n', q_min_90, q_max_200);
imprimir_resumo('Sol. A (q=105 kW/m2, w=44mm) ', met_A);
imprimir_resumo('Sol. B (w_L=60 mm, q=80kW/m2)', met_B);
imprimir_resumo('Sol. C (2 estagios + 60 mm)  ', met_C);
imprimir_resumo('Sol. D (Pre-aquec. T0=55 C)  ', met_D);
imprimir_resumo('Carenagem (h=100 W/(m^2.K))  ', met_h100);

% -------------------------------------------------------------------------
% 5. Graficos opcionais (ativar em MATLAB Desktop / sessao com display)
% -------------------------------------------------------------------------
gerar_figuras = false;
if gerar_figuras
    figure('Color', 'w');
    [TT, XX] = meshgrid(res_15C.t, res_15C.x);
    surf(XX, TT, res_15C.T, 'EdgeColor', 'none');
    colormap('jet'); colorbar;
    hold on;
    surf([0, p_fixos.L], [0, p_fixos.t_final], 90*ones(2,2), 'FaceAlpha', 0.22, 'FaceColor', 'b', 'EdgeColor', 'none');
    surf([0, p_fixos.L], [0, p_fixos.t_final], 200*ones(2,2), 'FaceAlpha', 0.20, 'FaceColor', 'r', 'EdgeColor', 'none');
    xlabel('Posicao x [m]'); ylabel('Tempo t [s]'); zlabel('Temperatura T [C]');
    xlim([0, p_fixos.L]); ylim([0, p_fixos.t_final]); zlim([20, 220]);
    xticks(0:0.01:0.06);
    xticklabels({'0,00', '0,01', '0,02', '0,03', '0,04', '0,05', '0,06'});
    view(-128, 25);
end
end

% =========================================================================
% Funcoes locais
% =========================================================================
function res = resolver_placa(h, q_flux, T_inf, N, Nt, w_laser, T_init, p)
    Bi_y = h * p.d / p.k;
    Bi   = h * (p.L^2) / (p.k * p.d);

    t_tilde_final = p.alpha * p.t_final / (p.L^2);
    t_tilde_L     = p.alpha * p.dt_laser / (p.L^2);
    dt_tilde      = t_tilde_final / Nt;
    dt_dim        = p.t_final / Nt;

    dx_tilde = 1.0 / (N + 1);
    x_int    = (1:N)' * dx_tilde;
    inv_dx2  = 1.0 / (dx_tilde^2);

    % Montagem da matriz tridiagonal esparsa A (Lomax & Pulliam, p. 44 em j=1)
    d_inf  = ones(N, 1) * inv_dx2;
    d_main = -2.0 * ones(N, 1) * inv_dx2;
    d_sup  = ones(N, 1) * inv_dx2;

    d_main(1) = (-2.0 / 3.0) * inv_dx2;
    d_sup(2)  = (2.0 / 3.0) * inv_dx2;

    A = spdiags([d_inf, d_main, d_sup], [-1, 0, 1], N, N);
    M = (1.0 + 2.0 * Bi * dt_tilde) * speye(N) - dt_tilde * A;

    % Fatoracao LU esparsa unica fora do laco temporal
    [L_lu, U_lu, P_lu, Q_lu] = lu(M);

    razao_laser = w_laser / p.w2;
    mask_laser  = double(x_int < razao_laser - 1e-12);

    if isnan(T_init)
        theta = zeros(N, 1);
    else
        theta = ones(N, 1) * ((T_init - T_inf) / T_inf);
    end

    x_full_tilde = [0.0; x_int; 1.0];
    x_full       = x_full_tilde * p.L;
    t_vec        = linspace(0.0, p.t_final, Nt + 1);
    T_hist       = zeros(N + 2, Nt + 1);

    theta_0      = (4.0 * theta(1) - theta(2)) / 3.0;
    T_hist(:, 1) = T_inf * (1.0 + [theta_0; theta; 0.0]);

    fator_q  = (p.L^2) / (p.k * p.d * T_inf);
    eh_func  = isa(q_flux, 'function_handle');

    for step = 1:Nt
        t_curr       = step * dt_dim;
        t_tilde_curr = step * dt_tilde;

        if eh_func
            q_curr = q_flux(t_curr);
        else
            if t_tilde_curr <= t_tilde_L + 1e-12
                q_curr = q_flux;
            else
                q_curr = 0.0;
            end
        end

        q_tilde_val = q_curr * fator_q;
        rhs         = theta + dt_tilde * (q_tilde_val * mask_laser);
        theta       = Q_lu * (U_lu \ (L_lu \ (P_lu * rhs)));

        theta_0            = (4.0 * theta(1) - theta(2)) / 3.0;
        T_hist(:, step + 1) = T_inf * (1.0 + [theta_0; theta; 0.0]);
    end

    res.x             = x_full;
    res.t             = t_vec;
    res.T             = T_hist;
    res.Bi_y          = Bi_y;
    res.Bi            = Bi;
    res.t_tilde_L     = t_tilde_L;
    res.t_tilde_final = t_tilde_final;
    if eh_func
        res.q_tilde = q_flux(0.0) * fator_q;
    else
        res.q_tilde = q_flux * fator_q;
    end
end

function met = calcular_metricas(res, p)
    x = res.x;
    t = res.t;
    T = res.T;

    x_borda    = p.w1 / 2.0;
    mask_x_pel = (x <= x_borda + 1e-12);

    nt         = length(t);
    T_centro_t = T(1, :);
    T_borda_t  = zeros(1, nt);
    T_max_pel  = zeros(1, nt);
    T_min_pel  = zeros(1, nt);

    for j = 1:nt
        Tb           = interp1(x, T(:, j), x_borda, 'linear');
        T_borda_t(j) = Tb;
        T_pel_j      = T(mask_x_pel, j);
        T_max_pel(j) = max(max(T_pel_j), Tb);
        T_min_pel(j) = min(min(T_pel_j), Tb);
    end

    mask_pulso = (t <= p.dt_laser + 1e-12);
    t_pulso    = t(mask_pulso);
    [~, idx_10] = min(abs(t - p.dt_laser));

    met.T_max_pulso     = max(T_max_pel(mask_pulso));
    met.T_centro_10s    = T_centro_t(idx_10);
    met.T_borda_10s     = T_borda_t(idx_10);
    met.delta_T_pel_10s = met.T_centro_10s - met.T_borda_10s;
    met.T_11mm_10s      = interp1(x, T(:, idx_10), 11.0e-3, 'linear');
    met.T_44mm_10s      = interp1(x, T(:, idx_10), 44.0e-3, 'linear');
    met.t_90_centro     = tempo_cruzamento(t_pulso, T_centro_t(mask_pulso), 90.0);
    met.t_90_borda      = tempo_cruzamento(t_pulso, T_min_pel(mask_pulso), 90.0);
    met.tempo_janela_pulso = duracao_janela(t_pulso, T_min_pel(mask_pulso), T_max_pel(mask_pulso), 90.0, 200.0);
    met.margem_200      = 200.0 - met.T_max_pulso;
end

function tc = tempo_cruzamento(t, y, alvo)
    idx = find(y >= alvo, 1, 'first');
    if isempty(idx)
        tc = NaN;
        return;
    elseif idx == 1
        tc = t(1);
        return;
    end
    y0 = y(idx - 1); y1 = y(idx);
    t0 = t(idx - 1); t1 = t(idx);
    tc = t0 + ((alvo - y0) / (y1 - y0)) * (t1 - t0);
end

function total = duracao_janela(t, T_min, T_max, T_lo, T_hi)
    total = 0.0;
    s = linspace(0.0, 1.0, 21);
    for i = 1:(length(t) - 1)
        dt = t(i + 1) - t(i);
        tmin_sub = T_min(i) + s * (T_min(i + 1) - T_min(i));
        tmax_sub = T_max(i) + s * (T_max(i + 1) - T_max(i));
        ok = double(tmin_sub >= T_lo & tmax_sub <= T_hi);
        frac = 0.5 * (ok(1:end-1) + ok(2:end));
        total = total + mean(frac) * dt;
    end
end

function imprimir_resumo(nome, m)
    fprintf('  %s | T_max=%.2f C | T_borda=%.2f C | dT=%.2f C | t90_b=%.2f s | Janela=%.2f s | Margem=%.2f C\n', ...
            nome, m.T_max_pulso, m.T_borda_10s, m.delta_T_pel_10s, m.t_90_borda, m.tempo_janela_pulso, m.margem_200);
end
