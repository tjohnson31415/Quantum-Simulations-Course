raw_corr = load('correlations.dat');

numPow = 6;
numTau = 20;
numBlocks = 20;

correlation_data = zeros(numPow, numPow, numTau, numBlocks);
for block = 1:numBlocks
    for tau =1:numTau
        correlation_data(:,:,tau,block) = utrivec2mat( raw_corr((block-1)*numTau + tau, :) );
    end
end

% The goal of the data, energies as a function of tau
%    -2 for the C_0 and t_0
energies = zeros(numTau - 2, numPow);
energiesBest = zeros(size(energies)); % best guess from all blocks
energiesErrors = zeros(size(energies));

% Loop over the different blocks that we have
for m = 1:numBlocks
    % Get the mean correlation data from the 1:m blocks
    corr = correlation_data( :, :, :, end-(m-1) );
    
    % Compute the eigenvalues as a function of tau, fist entry is C_0
    % Each column will contain the values of tau for one of the eigenvalues
    eigenvalues = zeros(numTau-1, numPow);
    for tau = 2 : numTau
        eigenvalues(tau-1, :) = eig(corr(:,:,tau), corr(:,:,2));
    end
    
    % Fill the energies array
    %   another tau value is used up as a starting point
    %   real to deal with i*pi values from a log(negative)
    for tau = 1 : numTau - 2
        energies(tau,:) = real( -log(eigenvalues(tau+1,:) ./ eigenvalues(tau,:)) );
    end
    
    % save the energies for the total data set and move on to less blocks
    if m == 1
        energiesBest = energies;
        continue;
    else
        energiesErrors = energiesErrors + (energies - energiesBest).^2;
    end
end

% Scale the errors from the number of blocks
energiesErrors = sqrt(energiesErrors .* ((numBlocks-1)/numBlocks));

errorbar(energiesBest(1:20,:), energiesErrors(1:20,:));
%plot(energiesBest(1:20,:))
xlabel('Correlation Time')
ylabel('Energy')
title('Energy Levels SHO (Npaths = 1mil)')
ylim([-.02 1])
xlim([0 20])
