%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function searchVisualWordIndex()
%  search the kd-tree visual word list, and locate the leaf node that query
%  point belongs
% z.li, started, 2010/04/05
% input:
%   q -  m x d query data points
%   indx - indx structure with dim and val of cuts
%   leafs - leaf nodes of offs of x. actually not needed
% output
%   nd  - m x1, the node numbers in the leafs, between 1:2^ht
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function [nd]=searchVisualWordIndex(q, indx, leafs)
function [nd]=searchVisualWordIndex(q, indx, leafs)
dbg = 0; 
if dbg
    x = randn(1024*16, 2); q = x(1:4,:); 
    ht = 10;
    [indx, leafs]=buildVisualWordList(x, ht); 
end

% const
dbgplot = 0;
if dbgplot == 1
    styls = ['.r'; '.b'; '.k'; '.m'; '+r'; '+b'; '+k'; '+m'; '*r'; '*b'; '*k'; '*m']; 
    roffs = randperm(12); styls = styls(roffs, :); 
end

% var
[m, kd]=size(q); 
nleaf = length(leafs); 
ht = fix(log2(nleaf)); 

if dbgplot
    plot(q(:,1), q(:,2), '*r'); hold on;
    for j=1:nleaf
        plot(x(leafs{j}, 1), x(leafs{j}, 2), '.'); hold on;
    end
    
end

for j=1:m
    % find where q(k) is
    k=1; qj = q(j,:); 
    % search thru cuts
    for h=1:ht
%         if dbgplot
%             fprintf('\n q(%d)=[%1.2f, %1.2f]', j, qj(1), qj(2)); 
%             fprintf(' q(j,%d) >< %1.2f', indx.d_cuts(k), indx.v_cuts(k)); 
%         end
        if qj(indx.d_cuts(k)) > indx.v_cuts(k) % right child
            k = 2*k + 1; 
        else  % left child
            k = 2*k;
        end
        % going down a level
        h = h + 1; 
    end % for h
    nd(j) = k - 2^ht + 1;
end

% find the leaf nodes
return;
