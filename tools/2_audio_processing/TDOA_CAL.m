%%
clc;clear;
batch = 1;
start_sample_list = [];
% exp 1: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
% exp 2: [1,2,3,4,5,6,7,8,9]
exp_time = [1];
for exp = exp_time
    disp(exp);
    sound_event = load("all_measurements\exp"+num2str(batch)+"\the pattern "+num2str(exp)+" sound seq.mat").seq_time;
    peak_count  = length(sound_event) -1;
    [master_data,fs] = audioread("audio/exp_"+num2str(batch)+"/master/ubuntu_"+num2str(exp)+".wav");
    [server1_data,fs]= audioread("audio/exp_"+num2str(batch)+"/server1/server1_"+num2str(exp)+".wav");
    [server2_data,fs]= audioread("audio/exp_"+num2str(batch)+"/server2/server2_"+num2str(exp)+".wav");
    [server3_data,fs]= audioread("audio/exp_"+num2str(batch)+"/server3/server3_"+num2str(exp)+".wav");

    for i =1:peak_count
        sound_record = get_duration(sound_event(i+1,:),sound_event(1,:));
        next_point = sound_record*fs;
        next_zone =    [floor(next_point - 0.4 * fs), ceil(next_point + 0.4 * fs)];

        [~,max_sig_master] = max(master_data(next_zone(1):next_zone(2), 1));
        [~,max_sig_server1] = max(server1_data(next_zone(1):next_zone(2), 1));
        [~,max_sig_server2] = max(server2_data(next_zone(1):next_zone(2), 1));
        [~,max_sig_server3] = max(server3_data(next_zone(1):next_zone(2), 1));

        start_sample_list = [start_sample_list; next_zone(1)+max_sig_master,next_zone(1)+max_sig_server1,next_zone(1)+max_sig_server2,next_zone(1)+max_sig_server3];
    end

    delay_mean  = zeros(peak_count,4);                          % TDOA measurement
    for i=1:peak_count
        start_sample = start_sample_list(i,:);

        zone_master  = [floor(start_sample(1)  - 0.3 * fs),ceil(start_sample(1) + 0.4 * fs)];
        zone_server1 = [floor(start_sample(2) - 0.3 * fs), ceil(start_sample(2) + 0.4 * fs)];
        zone_server2 = [floor(start_sample(3) - 0.3 * fs), ceil(start_sample(3) + 0.4 * fs)];
        zone_server3 = [floor(start_sample(4) - 0.3 * fs), ceil(start_sample(4) + 0.4 * fs)];

        master_sound_clip  = master_data(zone_master(1) :zone_master(2),:);
        server1_sound_clip = server1_data(zone_server1(1):zone_server1(2),:);
        server2_sound_clip = server2_data(zone_server2(1):zone_server2(2),:);
        server3_sound_clip = server3_data(zone_server3(1):zone_server3(2),:);
        
        delay_server1_master = array_delay_array(server1_sound_clip,master_sound_clip,zone_server1(1)-zone_master(1));
        delay_server2_master = array_delay_array(server2_sound_clip,master_sound_clip,zone_server2(1)-zone_master(1));
        delay_server3_master = array_delay_array(server3_sound_clip,master_sound_clip,zone_server3(1)-zone_master(1));


        master_channel_to_server1 = mean(delay_server1_master,2);
        master_channel_to_server2 = mean(delay_server2_master,2);
        master_channel_to_server3 = mean(delay_server3_master,2);

        master_to_server1  = mean(master_channel_to_server1);
        master_to_server2  = mean(master_channel_to_server2);
        master_to_server3  = mean(master_channel_to_server3);
        
        delay_mean(i,2) = master_to_server1;
        delay_mean(i,3) = master_to_server2;
        delay_mean(i,4) = master_to_server3;
    end
    save("all_measurements/exp"+num2str(batch)+"/1_TDOA/TDOA_exp_"+num2str(exp)+"mea.mat","delay_mean");
end

function [start_index,end_index] = find_start_end(data)
    fs = 16000;
    window_length = fs * 0.01;
    nfft = window_length;
    overlap = floor(nfft/8);
    [s, f, t] = spectrogram(data, tukeywin(window_length,0.25), overlap, nfft, fs);
    threshold = 1;
    start_time = t(find(max(abs(s)) > threshold, 1, 'first'));
    end_time = t(find(max(abs(s)) > 1.5, 1, 'last'));
    start_index = (start_time * fs);
    end_index = (end_time * fs);

end

function rotation_matrix = zyx_euler(z, y,x)
    z_matrix = [cos(z),-sin(z),0;
                sin(z),cos(z),0;
                0,0,1];
    y_matrix = [cos(y),0,sin(y);
                  0,1,    0;
            -sin(y),0,cos(y)];
    x_matrix = [1,0,0;
                0,cos(x),-sin(x);
                0,sin(x),cos(x)];
    rotation_matrix= z_matrix*y_matrix*x_matrix;
end

function delay_moment = array_delay_array(server_sound_clip,master_sound_clip,clip_diff) 
        % tau = server - master
        delay_moment = zeros(6,6);
        for main_channel = 1:6    
            for sub_channel = 1:6
                server_data = server_sound_clip(:,sub_channel);
                master_data = master_sound_clip(:,main_channel);
                max_len = max(length(server_data), length(master_data));
                server_data = [server_data;zeros(max_len-length(server_data),1)];
                master_data = [master_data;zeros(max_len-length(master_data),1)];
                [tau,R,lag]  = gccphat(server_data,master_data);
                delay_moment(main_channel,sub_channel) = (clip_diff+tau);

            end
        end
end

function duration = get_duration(current_time,start_time) 
    current_time= current_time(2)+current_time(3)/1e9;
    start_time = start_time(2)+start_time(3)/1e9;
    duration = current_time - start_time;
end
