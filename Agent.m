classdef Agent
    %% Định nghĩa các thuộc tính
    properties
        StateSize        % Kích thước của trạng thái
        ActionSize       % Kích thước của hành động
        FcSize           % Kích thước lớp của mạng nơ ron
        ActorWeights     % Trọng số của mạng Actor
        CriticQ1Weights  % Trọng số của mạng Critic Q1
        CriticQ2Weights  % Trọng số của mạng Critic Q2
        TargetQ1Weights  % Trọng số của mạng Target Q1
        TargetQ2Weights  % Trọng số của mạng Target Q2
        Actor_lrate      % Tốc độ học cho Actor
        Critic_lrate     % Tốc độ học cho Critic
        Entropy_lrate    % Tốc độ học cho Entropy
        Gamma            % Hệ số chiết khấu
        Tau              % Hệ số cập nhật Target
        Alpha            % Hệ số Entropy
        AlphaWeights     % m_t và v_t của Alpha
        H                % Target Entropy
        Beta1            % Hệ số Beta 1 trong Adam
        Beta2            % Hệ số Beta 2 trong Adam
        l2               % Hệ số L2 Regularization
    end
    %% Định nghĩa các phương thức
    methods
        %% Phương thức để khởi tạo
        function obj = Agent(state_size, action_size, fc_size, actor_lrate, critic_lrate, entropy_lrate, gamma, tau)
            if nargin > 0
                obj.StateSize = state_size;
                obj.ActionSize = action_size;
                obj.FcSize = fc_size;
                obj.ActorWeights = KhoiTaoActor(state_size, fc_size, action_size);
                obj.CriticQ1Weights = KhoiTaoCritic(state_size, fc_size, action_size);
                obj.CriticQ2Weights = KhoiTaoCritic(state_size, fc_size, action_size);
                for i = 1:10
                    obj.TargetQ1Weights{i} = obj.CriticQ1Weights{i};
                    obj.TargetQ2Weights{i} = obj.CriticQ2Weights{i};
                end
                obj.Actor_lrate = actor_lrate;
                obj.Critic_lrate = critic_lrate;
                obj.Entropy_lrate = entropy_lrate;
                obj.Gamma = gamma;
                obj.Tau = tau;
                obj.Alpha = 0.1;
                obj.AlphaWeights = [0, 0];
                obj.H = -action_size;
                obj.Beta1 = 0.9;
                obj.Beta2 = 0.999;
                obj.l2 = 0.0001;
            end
        end
        %% Phương thức để chọn hành động dựa trên trạng thái hiện tại
        function [action, logProb] = selectAction(obj, state)
            [mu, sigma, ~] = obj.actorForward(obj.ActorWeights, state);
            epsilon = randn(obj.ActionSize, size(state, 2));
            repara = mu + sigma.*epsilon;
            action = 0.5*tanh(repara) + 0.5;
            logProb = log(2/sqrt(2*pi)) - 0.5*epsilon(1, :).^2 - log(sigma(1, :)) - 2*(log(2) - repara(1, :) - log(1 + exp(-2*repara(1, :)))) ...
                    + log(2/sqrt(2*pi)) - 0.5*epsilon(2, :).^2 - log(sigma(2, :)) - 2*(log(2) - repara(2, :) - log(1 + exp(-2*repara(2, :))));
        end
        %% Phương thức để cập nhật Critic Q1
        function obj = updateCriticQ1(obj, batch)
            % Chia batch
            batch_size = size(batch, 2);
            state = batch(1:obj.StateSize, :);
            action = batch(obj.StateSize + 1:obj.StateSize + obj.ActionSize, :);
            reward = batch(obj.StateSize + obj.ActionSize + 1, :);
            nextState = batch(obj.StateSize + obj.ActionSize + 1 + 1:obj.StateSize + obj.ActionSize + 1 + obj.StateSize, :);
            isDone = batch(obj.StateSize + obj.ActionSize + 1 + obj.StateSize + 1, :);
            % Tính Q(a_t, s_t)
            [CriticQ1, CriticQ1Layers] = obj.criticForward(obj.CriticQ1Weights, state, action);
            % Tính Q(s_t+1, a_t+1) và log(pi(a_t+1|s_t+1)
            [nextAction, nextLogPi] = selectAction(obj, nextState);
            [nextTargetQ1, ~] = obj.criticForward(obj.TargetQ1Weights, nextState, nextAction);
            % Tách trọng số
            fc_body_Weights = obj.CriticQ1Weights{5}; 
            critic_Weights = obj.CriticQ1Weights{7};
            concat1 = obj.CriticQ1Weights{9};
            concat2 = obj.CriticQ1Weights{10};
            % Tách layers
            concat = CriticQ1Layers{3};
            relu_body = CriticQ1Layers{4};
            fc_body = CriticQ1Layers{5};
            body_output = CriticQ1Layers{6};
            % Tính Gradient Critic
            dJ_dcritic = (1/batch_size)*(CriticQ1 - (reward + obj.Gamma*(1 - isDone).*(nextTargetQ1 - obj.Alpha*nextLogPi)));
            dJ_dcritic_Weights = dJ_dcritic*body_output';
            dJ_dcritic_Bias = sum(dJ_dcritic, 2);
            dJ_dbody_output = critic_Weights'*dJ_dcritic;
            dJ_dfc_body = dJ_dbody_output.*heaviside(fc_body);
            dJ_dfc_body_Weights = dJ_dfc_body*relu_body';
            dJ_dfc_body_Bias = sum(dJ_dfc_body, 2);
            dJ_drelu_body = fc_body_Weights'*dJ_dfc_body;
            dJ_dconcat = dJ_drelu_body.*heaviside(concat);
            dJ_dfc_2 = concat2'*dJ_dconcat;
            dJ_dfc_2_Weights = dJ_dfc_2*action';
            dJ_dfc_2_Bias = sum(dJ_dfc_2, 2);
            dJ_dfc_1 = concat1'*dJ_dconcat;
            dJ_dfc_1_Weights = dJ_dfc_1*state';
            dJ_dfc_1_Bias = sum(dJ_dfc_1, 2);
            % Gán
            dweights{1} = dJ_dfc_1_Weights + obj.l2*obj.CriticQ1Weights{1};
            dweights{2} = dJ_dfc_1_Bias + obj.l2*obj.CriticQ1Weights{2};
            dweights{3} = dJ_dfc_2_Weights + obj.l2*obj.CriticQ1Weights{3};
            dweights{4} = dJ_dfc_2_Bias + obj.l2*obj.CriticQ1Weights{4};
            dweights{5} = dJ_dfc_body_Weights + obj.l2*obj.CriticQ1Weights{5};
            dweights{6} = dJ_dfc_body_Bias + obj.l2*obj.CriticQ1Weights{6};
            dweights{7} = dJ_dcritic_Weights + obj.l2*obj.CriticQ1Weights{7};
            dweights{8} = dJ_dcritic_Bias + obj.l2*obj.CriticQ1Weights{8};
            % Cập nhật trọng số adam
            for i = 1:8
                % Cập nhật m_t
                obj.CriticQ1Weights{i + 10} = obj.Beta1*obj.CriticQ1Weights{i + 10} + (1 - obj.Beta1)*dweights{i};
                % Cập nhật v_t
                obj.CriticQ1Weights{i + 18} = obj.Beta2*obj.CriticQ1Weights{i + 18} + (1 - obj.Beta2)*(dweights{i}.^2);
                % Tính gradient step
                gradThreshold = inf;
                gradStep = obj.CriticQ1Weights{i + 10}./(sqrt(obj.CriticQ1Weights{i + 18}) + 1e-8);
                if norm(gradStep) > gradThreshold
                    gradStep = gradThreshold*gradStep/norm(gradStep);
                end
                % Cập nhật
                obj.CriticQ1Weights{i} = obj.CriticQ1Weights{i} - obj.Critic_lrate*gradStep;
            end
        end
        %% Phương thức để cập nhật Critic Q2
        function obj = updateCriticQ2(obj, batch)
            % Chia batch
            batch_size = size(batch, 2);
            state = batch(1:obj.StateSize, :);
            action = batch(obj.StateSize + 1:obj.StateSize + obj.ActionSize, :);
            reward = batch(obj.StateSize + obj.ActionSize + 1, :);
            nextState = batch(obj.StateSize + obj.ActionSize + 1 + 1:obj.StateSize + obj.ActionSize + 1 + obj.StateSize, :);
            isDone = batch(obj.StateSize + obj.ActionSize + 1 + obj.StateSize + 1, :);
            % Tính Q(a_t, s_t)
            [CriticQ2, CriticQ2Layers] = obj.criticForward(obj.CriticQ2Weights, state, action);
            % Tính Q(s_t+1, a_t+1) và log(pi(a_t+1|s_t+1)
            [nextAction, nextLogPi] = selectAction(obj, nextState);
            [nextTargetQ2, ~] = obj.criticForward(obj.TargetQ2Weights, nextState, nextAction);
            % Tách trọng số
            fc_body_Weights = obj.CriticQ2Weights{5}; 
            critic_Weights = obj.CriticQ2Weights{7};
            concat1 = obj.CriticQ2Weights{9};
            concat2 = obj.CriticQ2Weights{10};
            % Tách layers
            concat = CriticQ2Layers{3};
            relu_body = CriticQ2Layers{4};
            fc_body = CriticQ2Layers{5};
            body_output = CriticQ2Layers{6};
            % Tính Gradient Critic
            dJ_dcritic = (1/batch_size)*(CriticQ2 - (reward + obj.Gamma*(1 - isDone).*(nextTargetQ2 - obj.Alpha*nextLogPi)));
            dJ_dcritic_Weights = dJ_dcritic*body_output';
            dJ_dcritic_Bias = sum(dJ_dcritic, 2);
            dJ_dbody_output = critic_Weights'*dJ_dcritic;
            dJ_dfc_body = dJ_dbody_output.*heaviside(fc_body);
            dJ_dfc_body_Weights = dJ_dfc_body*relu_body';
            dJ_dfc_body_Bias = sum(dJ_dfc_body, 2);
            dJ_drelu_body = fc_body_Weights'*dJ_dfc_body;
            dJ_dconcat = dJ_drelu_body.*heaviside(concat);
            dJ_dfc_2 = concat2'*dJ_dconcat;
            dJ_dfc_2_Weights = dJ_dfc_2*action';
            dJ_dfc_2_Bias = sum(dJ_dfc_2, 2);
            dJ_dfc_1 = concat1'*dJ_dconcat;
            dJ_dfc_1_Weights = dJ_dfc_1*state';
            dJ_dfc_1_Bias = sum(dJ_dfc_1, 2);
            % Gán
            dweights{1} = dJ_dfc_1_Weights + obj.l2*obj.CriticQ2Weights{1};
            dweights{2} = dJ_dfc_1_Bias + obj.l2*obj.CriticQ2Weights{2};
            dweights{3} = dJ_dfc_2_Weights + obj.l2*obj.CriticQ2Weights{3};
            dweights{4} = dJ_dfc_2_Bias + obj.l2*obj.CriticQ2Weights{4};
            dweights{5} = dJ_dfc_body_Weights + obj.l2*obj.CriticQ2Weights{5};
            dweights{6} = dJ_dfc_body_Bias + obj.l2*obj.CriticQ2Weights{6};
            dweights{7} = dJ_dcritic_Weights + obj.l2*obj.CriticQ2Weights{7};
            dweights{8} = dJ_dcritic_Bias + obj.l2*obj.CriticQ2Weights{8};
            % Cập nhật trọng số adam
            for i = 1:8
                % Cập nhật m_t
                obj.CriticQ2Weights{i + 10} = obj.Beta1*obj.CriticQ2Weights{i + 10} + (1 - obj.Beta1)*dweights{i};
                % Cập nhật v_t
                obj.CriticQ2Weights{i + 18} = obj.Beta2*obj.CriticQ2Weights{i + 18} + (1 - obj.Beta2)*(dweights{i}.^2);
                % Tính gradient step
                gradThreshold = inf;
                gradStep = obj.CriticQ2Weights{i + 10}./(sqrt(obj.CriticQ2Weights{i + 18}) + 1e-8);
                if norm(gradStep) > gradThreshold
                    gradStep = gradThreshold*gradStep/norm(gradStep);
                end
                % Cập nhật
                obj.CriticQ2Weights{i} = obj.CriticQ2Weights{i} - obj.Critic_lrate*gradStep;
            end
        end
        %% Phương thức để cập nhật Actor
        function obj = updateActor(obj, batch)
            % Chia batch
            batch_size = size(batch, 2);
            state = batch(1:obj.StateSize, :);
            % Tính mu, sigma và epsilon từ batch
            epsilon = randn(obj.ActionSize, size(state, 2));
            [mu, sigma, ActorLayers] = obj.actorForward(obj.ActorWeights, state);
            repara = mu + sigma.*epsilon;
            action = 0.5*tanh(repara) + 0.5;
            % Tính đạo hàm của log(pi(a_t|s_t) theo mu và sigma
            dlog_dfc_mean = 4*action - 2;
            dlog_dstd = (4*action - 2).*epsilon - 1./sigma;
            % Tính đạo hàm của Q(s_t, a_t) theo a_t
            dcritic_da = zeros(size(action));
            for i = 1:batch_size
                % Chọn Q min
                [CriticQ1, CriticQ1Layers] = obj.criticForward(obj.TargetQ1Weights, state(:, i), action(:, i));
                [CriticQ2, CriticQ2Layers] = obj.criticForward(obj.TargetQ2Weights, state(:, i), action(:, i));
                if CriticQ1 < CriticQ2
                    weights = obj.TargetQ1Weights;
                    layers = CriticQ1Layers;
                else
                    weights = obj.TargetQ2Weights;
                    layers = CriticQ2Layers;
                end
                % Tách trọng số
                fc_2_Weights = weights{3};
                fc_body_Weights = weights{5};
                critic_Weights = weights{7};
                concat2 = weights{10};
                % Tách layers
                concat = layers{3};
                fc_body = layers{5};
                % Tính đạo hàm
                dcritic_da(:, i) = (critic_Weights*diag(heaviside(fc_body))*fc_body_Weights*diag(heaviside(concat))*concat2*fc_2_Weights)';
            end
            % Tính đạo hàm của a_t theo mu và sigma
            da_dfc_mean = 0.5*(1 - (2*action - 1).^2);
            da_dstd = 0.5*(1 - (2*action - 1).^2).*epsilon;
            % Tách trọng số
            fc_body_Weights = obj.ActorWeights{3};
            fc_mean_Weights = obj.ActorWeights{5};
            fc_std_Weights = obj.ActorWeights{7};
            % Tách layers
            fc_1 = ActorLayers{1};
            relu_body = ActorLayers{2};
            fc_body = ActorLayers{3};
            body_output = ActorLayers{4};
            fc_std = ActorLayers{6};
            % Tính Gradient Actor
            dJ_dfc_mean = (1/batch_size)*(obj.Alpha*dlog_dfc_mean - dcritic_da.*da_dfc_mean);
            dJ_dstd = (1/batch_size)*(obj.Alpha*dlog_dstd - dcritic_da.*da_dstd);
            dJ_dfc_std = dJ_dstd.*sigmoid(fc_std);
            dJ_dfc_std_Weights = dJ_dfc_std*body_output';
            dJ_dfc_std_Bias = sum(dJ_dfc_std, 2);
            dJ_dfc_mean_Weights = dJ_dfc_mean*body_output';
            dJ_dfc_mean_Bias = sum(dJ_dfc_mean, 2);
            dJ_dbody_output = fc_mean_Weights'*dJ_dfc_mean + fc_std_Weights'*dJ_dfc_std;
            dJ_dfc_body = dJ_dbody_output.*heaviside(fc_body);
            dJ_dfc_body_Weights = dJ_dfc_body*relu_body';
            dJ_dfc_body_Bias = sum(dJ_dfc_body, 2);
            dJ_drelu_body = fc_body_Weights'*dJ_dfc_body;
            dJ_dfc_1 = dJ_drelu_body.*heaviside(fc_1);
            dJ_dfc_1_Weights = dJ_dfc_1*state';
            dJ_dfc_1_Bias = sum(dJ_dfc_1, 2);
            % Gán
            dweights{1} = dJ_dfc_1_Weights + obj.l2*obj.ActorWeights{1};
            dweights{2} = dJ_dfc_1_Bias + obj.l2*obj.ActorWeights{2};
            dweights{3} = dJ_dfc_body_Weights + obj.l2*obj.ActorWeights{3};
            dweights{4} = dJ_dfc_body_Bias + obj.l2*obj.ActorWeights{4};
            dweights{5} = dJ_dfc_mean_Weights + obj.l2*obj.ActorWeights{5};
            dweights{6} = dJ_dfc_mean_Bias + obj.l2*obj.ActorWeights{6};
            dweights{7} = dJ_dfc_std_Weights + obj.l2*obj.ActorWeights{7};
            dweights{8} = dJ_dfc_std_Bias + obj.l2*obj.ActorWeights{8};
            % Cập nhật trọng số adam
            for i = 1:8
                % Cập nhật m_t
                obj.ActorWeights{i + 8} = obj.Beta1*obj.ActorWeights{i + 8} + (1 - obj.Beta1)*dweights{i};
                % Cập nhật v_t
                obj.ActorWeights{i + 16} = obj.Beta2*obj.ActorWeights{i + 16} + (1 - obj.Beta2)*(dweights{i}.^2);
                % Tính gradient step
                gradThreshold = inf;
                gradStep = obj.ActorWeights{i + 8}./(sqrt(obj.ActorWeights{i + 16}) + 1e-8);
                if norm(gradStep) > gradThreshold
                    gradStep = gradThreshold*gradStep/norm(gradStep);
                end
                % Cập nhật
                obj.ActorWeights{i} = obj.ActorWeights{i} - obj.Actor_lrate*gradStep;
            end
        end
        %% Phương thức để cập nhật Temperature
        function obj = updateTemperature(obj, batch)
            % Chia batch
            state = batch(1:obj.StateSize, :);
            [~, logProb] = selectAction(obj, state);
            dAlpha = mean(- logProb - obj.H);
            % Cập nhật m_t
            obj.AlphaWeights(1) = obj.Beta1*obj.AlphaWeights(1) + (1 - obj.Beta1)*dAlpha;
            % Cập nhật v_t
            obj.AlphaWeights(2) = obj.Beta2*obj.AlphaWeights(2) + (1 - obj.Beta2)*(dAlpha^2);
            % Tính gradient step
            gradThreshold = inf;
            gradStep = obj.AlphaWeights(1)./(sqrt(obj.AlphaWeights(2)) + 1e-8);
            if norm(gradStep) > gradThreshold
                gradStep = gradThreshold*gradStep/norm(gradStep);
            end
            % Cập nhật
            obj.Alpha = obj.Alpha - obj.Entropy_lrate*gradStep;
        end
        %% Phương thức để cập nhật Target Q1
        function obj = updateTargetQ1(obj)
            for i = 1:8
                obj.TargetQ1Weights{i} = obj.Tau*obj.CriticQ1Weights{i} + (1 - obj.Tau)*obj.TargetQ1Weights{i};
            end
        end
        %% Phương thức để cập nhật Target Q1
        function obj = updateTargetQ2(obj)
            for i = 1:8
                obj.TargetQ2Weights{i} = obj.Tau*obj.CriticQ2Weights{i} + (1 - obj.Tau)*obj.TargetQ2Weights{i};
            end
        end
        %% Phương thức tính forward cho Actor
        function [mu, sigma, layers] = actorForward(~, weights, state)
            % Tách trọng số
            fc_1_Weights = weights{1};
            fc_1_Bias = weights{2};
            fc_body_Weights = weights{3};
            fc_body_Bias = weights{4};
            fc_mean_Weights = weights{5};
            fc_mean_Bias = weights{6};
            fc_std_Weights = weights{7};
            fc_std_Bias = weights{8};
            % Tính toán các layers
            layers{1} = fc_1_Weights*state + fc_1_Bias; % fc_1
            layers{2} = max(0, layers{1}); % relu_body
            layers{3} = fc_body_Weights*layers{2} + fc_body_Bias; % fc_body
            layers{4} = max(0, layers{3}); % body_output
            layers{5} = fc_mean_Weights*layers{4} + fc_mean_Bias; % fc_mean
            layers{6} = fc_std_Weights*layers{4} + fc_std_Bias; % fc_std
            layers{7} = log(exp(layers{6}) + 1); % std
            mu = layers{5};
            sigma = layers{7};
        end
        %% Phương thức tính forward cho Critic
        function [Q, layers] = criticForward(~, weights, state, action)
            % Tách trọng số
            fc_1_Weights = weights{1};
            fc_1_Bias = weights{2};
            fc_2_Weights = weights{3};
            fc_2_Bias = weights{4};
            fc_body_Weights = weights{5};
            fc_body_Bias = weights{6};
            critic_Weights = weights{7};
            critic_Bias = weights{8};
            concat1 = weights{9};
            concat2 = weights{10};
            % Tính toán các layers
            layers{1} = fc_1_Weights*state + fc_1_Bias; % fc_1
            layers{2} = fc_2_Weights*action + fc_2_Bias; % fc_2
            layers{3} = concat1*layers{1} + concat2*layers{2}; % concat
            layers{4} = max(0, layers{3}); % relu_body
            layers{5} = fc_body_Weights*layers{4} + fc_body_Bias; % fc_body
            layers{6}= max(0, layers{5}); % body_output
            layers{7} = critic_Weights*layers{6} + critic_Bias; % critic
            Q = layers{7};
        end
    end
end
%% Hàm khởi tạo trọng mạng Actor
function weights = KhoiTaoActor(state_size, fc_size, action_size)
    weights{1} = (1/sqrt(fc_size))*randn(fc_size, state_size); % fc_1_Weights
    weights{2} = zeros(fc_size, 1); % fc_1_Bias
    weights{3} = (1/sqrt(fc_size))*randn(fc_size, fc_size); % fc_body_Weights
    weights{4} = zeros(fc_size, 1); % fc_body_Bias
    weights{5} = (1/sqrt(fc_size))*randn(action_size, fc_size); % fc_mean_Weights
    weights{6} = zeros(action_size, 1); % fc_mean_Bias
    weights{7} = (1/sqrt(fc_size))*randn(action_size, fc_size); % fc_std_Weights
    weights{8} = zeros(action_size, 1); % fc_std_Bias
    for i = 1:8
        weights{i + 8} = zeros(size(weights{i})); % m_t
        weights{i + 16} = zeros(size(weights{i})); % v_t
    end
end
%% Hàm khởi tạo trọng mạng Critic
function weights = KhoiTaoCritic(state_size, fc_size, action_size)
    weights{1} = (1/sqrt(fc_size))*randn(fc_size, state_size); % fc_1_Weights
    weights{2} = zeros(fc_size, 1); % fc_1_Bias
    weights{3} = (1/sqrt(fc_size))*randn(fc_size, action_size); % fc_2_Weights
    weights{4} = zeros(fc_size, 1); % fc_2_Bias
    weights{5} = (1/sqrt(fc_size))*randn(fc_size, 2*fc_size); % fc_body_Weights
    weights{6} = zeros(fc_size, 1); % fc_body_Bias
    weights{7} = (1/sqrt(fc_size))*randn(1, fc_size); % critic_Weights
    weights{8} = zeros(1, 1); % critic_Bias
    weights{9} = [diag(ones(fc_size, 1)); zeros(fc_size, fc_size)]; % concat1
    weights{10} = [zeros(fc_size, fc_size); diag(ones(fc_size, 1))]; % concat2
    for i = 1:8
        weights{i + 10} = zeros(size(weights{i})); % m_t
        weights{i + 18} = zeros(size(weights{i})); % v_t
    end
end
%% Hàm sigmoid
function y = sigmoid(x)
    y = 1./(1 + exp(-x));
end