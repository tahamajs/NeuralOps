#!/usr/bin/env perl
use strict;
use warnings;
use utf8;
binmode(STDOUT, ":utf8");

my $root_dir = "academic/paper/arxiv/monograph_blueprint/appendix_c_203module_architectural_catalog/batch05_gla_hgrn";
my $out_file = "academic/paper/arxiv/sections/appendix_c_batches/batch05_gla_hgrn.tex";

my @modules = (
    ["042", "gated_linear_attn", "Gated Linear Attn", "$root_dir/gated_linear_attn_SPEC.md"],
    ["043", "gla_attention", "Gla Attention", "$root_dir/gla_attention_SPEC.md"],
    ["044", "goldfinch_ahn", "Goldfinch Ahn", "$root_dir/goldfinch_ahn_SPEC.md"],
    ["045", "goldfinch_attention", "Goldfinch Attention", "$root_dir/goldfinch_attention_SPEC.md"],
    ["046", "google_linear_attention", "Google Linear Attention", "$root_dir/google_linear_attention_SPEC.md"],
    ["047", "gqa_attention", "Gqa Attention", "$root_dir/gqa_attention_SPEC.md"],
    ["048", "griffin_recurrence", "Griffin Recurrence", "$root_dir/griffin_recurrence_SPEC.md"],
    ["049", "hgrn2_attention", "Hgrn2 Attention", "$root_dir/hgrn2_attention_SPEC.md"],
    ["050", "hgrn2_gated_deltanet", "Hgrn2 Gated Deltanet", "$root_dir/hgrn2_gated_deltanet_SPEC.md"],
    ["051", "hgrn2_gated_layer", "Hgrn2 Gated Layer", "$root_dir/hgrn2_gated_layer_SPEC.md"],
);

sub format_text {
    my ($text) = @_;
    
    # Split by math tokens: $$...$$ or $...$
    my @parts = split(/(\$\$[^\$]+\$\$|\$[^\$]+\$)/s, $text);
    my $result = "";
    
    for my $p (@parts) {
        if ($p =~ /^\$/) {
            # Math token - keep as is
            $result .= $p;
        } else {
            # Prose token
            my $s = $p;
            
            # Replace code `...` with breakable tokens
            $s =~ s/`([^`]+)`/my $c = $1; $c =~ s{_}{\\_\\allowbreak }g; $c =~ s{\.}{.\\allowbreak }g; $c =~ s{\/}{\/\\allowbreak }g; $c =~ s{, }{, \\allowbreak }g; $c =~ s{=}{=\\allowbreak }g; $c =~ s{%}{\\%}g; $c =~ s{&}{\\&}g; $c =~ s{#}{\\#}g; "\\texttt{$c}"/ge;
            
            # Replace bold **...**
            $s =~ s/\*\*([^*]+)\*\*/\\textbf{$1}/g;
            
            # Replace italic *...* (not preceded or followed by *)
            $s =~ s/(?<!\*)\*([^*\n]+)\*(?!\*)/\\emph{$1}/g;
            
            # Escape %, &, #, _
            $s =~ s/(?<!\\)%/\\%/g;
            $s =~ s/(?<!\\)&/\\&/g;
            $s =~ s/(?<!\\)#/\\#/g;
            $s =~ s/(?<!\\)_/\\_/g;
            
            $result .= $s;
        }
    }
    return $result;
}

sub process_spec_file {
    my ($mod_idx, $mod_key, $mod_title, $filepath) = @_;
    
    open my $fh, '<:encoding(UTF-8)', $filepath or die "Cannot open $filepath: $!";
    my @lines = <$fh>;
    close $fh;
    
    my @out;
    my $escaped_key = $mod_key;
    $escaped_key =~ s/_/\\_/g;
    
    push @out, "% --- Module $mod_idx: $mod_key ---\n";
    push @out, "\\subsubsection*{Module $mod_idx: $mod_title (\\texorpdfstring{\\texttt{$escaped_key}}{$mod_key})}\n";
    push @out, "\\label{sec:mod_$mod_key}\n\n";
    
    my $in_code = 0;
    my $code_lang = "";
    my @code_lines = ();
    my $in_table = 0;
    my @table_rows = ();
    my $in_math = 0;
    my @math_lines = ();
    my $current_section_title = "";
    
    my sub flush_table {
        return unless @table_rows;
        
        my @parsed;
        for my $r (@table_rows) {
            my $line = $r;
            $line =~ s/^\s*\|\s*//;
            $line =~ s/\s*\|\s*$//;
            my @cells = split(/\s*\|\s*/, $line);
            push @parsed, \@cells;
        }
        @table_rows = ();
        $in_table = 0;
        
        return if @parsed < 2;
        
        my @data_rows;
        my $headers = $parsed[0];
        for my $r (@parsed[1..$#parsed]) {
            my $row_str = join(" ", @$r);
            next if $row_str =~ /^\s*:-/;
            push @data_rows, $r;
        }
        
        my $num_cols = scalar(@$headers);
        for my $r (@data_rows) {
            $num_cols = scalar(@$r) if scalar(@$r) > $num_cols;
        }
        
        my $col_spec = "l" . ("c" x ($num_cols - 1));
        
        push @out, "\\begin{center}\n";
        push @out, "\\resizebox{\\linewidth}{!}{%\n";
        push @out, "\\begin{tabular}{$col_spec}\n";
        push @out, "\\toprule\n";
        
        my @fmt_headers = map { "\\textbf{" . format_text($_) . "}" } @$headers;
        push @out, join(" & ", @fmt_headers) . " \\\\\n";
        push @out, "\\midrule\n";
        
        for my $row (@data_rows) {
            my @cells = @$row;
            while (scalar(@cells) < $num_cols) { push @cells, ""; }
            my @fmt_cells = map { format_text($_) } @cells;
            push @out, join(" & ", @fmt_cells) . " \\\\\n";
        }
        
        push @out, "\\bottomrule\n";
        push @out, "\\end{tabular}%\n";
        push @out, "}\n";
        push @out, "\\end{center}\n\n";
    };
    
    for my $line (@lines) {
        chomp $line;
        
        # Handle code blocks
        if ($line =~ /^```(.*)$/) {
            if (!$in_code) {
                flush_table() if $in_table;
                $in_code = 1;
                $code_lang = $1;
                $code_lang =~ s/^\s+|\s+$//g;
                @code_lines = ();
            } else {
                $in_code = 0;
                my $full_code = join("\n", @code_lines);
                
                if ($code_lang =~ /latex/i && $full_code =~ /\\begin\{tikzpicture\}/) {
                    my $fig_caption = $current_section_title || "$mod_title Architectural Diagram";
                    $fig_caption =~ s/^\d+\.\d+\s*//;
                    my $fmt_caption = format_text($fig_caption);
                    
                    push @out, "\\begin{figure}[htbp]\n";
                    push @out, "\\centering\n";
                    push @out, "\\resizebox{0.90\\linewidth}{!}{%\n";
                    push @out, $full_code . "\n";
                    push @out, "}%\n";
                    push @out, "\\caption{$fmt_caption}\n";
                    push @out, "\\label{fig:mod_${mod_key}_arch}\n";
                    push @out, "\\end{figure}\n\n";
                } elsif ($code_lang =~ /python/i) {
                    push @out, "\\begin{lstlisting}[language=Python, style=systemscode]\n";
                    push @out, $full_code . "\n";
                    push @out, "\\end{lstlisting}\n\n";
                } elsif ($code_lang =~ /(c\+\+|cpp|cuda)/i) {
                    push @out, "\\begin{lstlisting}[language=C++, style=systemscode]\n";
                    push @out, $full_code . "\n";
                    push @out, "\\end{lstlisting}\n\n";
                } else {
                    push @out, "\\begin{lstlisting}[style=systemscode]\n";
                    push @out, $full_code . "\n";
                    push @out, "\\end{lstlisting}\n\n";
                }
                @code_lines = ();
            }
            next;
        }
        
        if ($in_code) {
            push @code_lines, $line;
            next;
        }
        
        # Handle display math $$
        my $trim_line = $line;
        $trim_line =~ s/^\s+//;
        if ($trim_line =~ /^\$\$/) {
            flush_table() if $in_table;
            if ($trim_line eq '$$') {
                if (!$in_math) {
                    $in_math = 1;
                    @math_lines = ();
                } else {
                    $in_math = 0;
                    my $math_content = join("\n", @math_lines);
                    push @out, "\\begin{equation}\n";
                    push @out, $math_content . "\n";
                    push @out, "\\end{equation}\n\n";
                    @math_lines = ();
                }
                next;
            } else {
                # Single line $$...$$
                my $m = $trim_line;
                $m =~ s/^\$\$\s*//;
                $m =~ s/\s*\$\$$//;
                push @out, "\\begin{equation}\n";
                push @out, $m . "\n";
                push @out, "\\end{equation}\n\n";
                next;
            }
        }
        
        if ($in_math) {
            push @math_lines, $line;
            next;
        }
        
        # Handle table rows
        if ($line =~ /^\s*\|.*\|\s*$/) {
            $in_table = 1;
            push @table_rows, $line;
            next;
        } else {
            flush_table() if $in_table;
        }
        
        # Skip top markdown header `# Technical Specification...`
        if ($line =~ /^#\s+/) {
            next;
        }
        
        # Handle horizontal divider `---`
        if ($line =~ /^---/) {
            push @out, "\\medskip\\hrule\\medskip\n";
            next;
        }
        
        # Handle H2 `## 1. Executive Identification...`
        if ($line =~ /^##\s+(.*)$/) {
            my $h2 = $1;
            $h2 =~ s/^\s+|\s+$//g;
            my $fmt_h2 = format_text($h2);
            push @out, "\\medskip\\hrule\\medskip\n";
            push @out, "\\paragraph*{\\textbf{$fmt_h2}}\n";
            next;
        }
        
        # Handle H3 `### 1.1 Canonical Registry & Identity`
        if ($line =~ /^###\s+(.*)$/) {
            my $h3 = $1;
            $h3 =~ s/^\s+|\s+$//g;
            $current_section_title = $h3;
            my $fmt_h3 = format_text($h3);
            push @out, "\\noindent\\textbf{$fmt_h3}\\par\\smallskip\n";
            next;
        }
        
        # Handle H4 `#### Autograd VJP...`
        if ($line =~ /^####\s+(.*)$/) {
            my $h4 = $1;
            $h4 =~ s/^\s+|\s+$//g;
            my $fmt_h4 = format_text($h4);
            push @out, "\\noindent\\textbf{$fmt_h4}\\par\\smallskip\n";
            next;
        }
        
        # Handle metadata bullets before section 1
        if ($line =~ /^- \*\*Target Location\*\*:\s*`([^`]+)`/) {
            push @out, "\\textbf{--} \\textbf{Catalog Dossier}: Appendix C, Batch 05 (GLA \\& HGRN2) / Module $mod_idx\\\\\n";
            next;
        }
        
        # Handle list items
        if ($line =~ /^\s*-\s+(.*)$/) {
            my $item = $1;
            push @out, "\\textbf{--} " . format_text($item) . "\\\\\n";
            next;
        }
        
        # Handle numbered lists
        if ($line =~ /^\s*(\d+)\.\s+(.*)$/) {
            my $num = $1;
            my $item = $2;
            push @out, "\\textbf{$num.}\\ " . format_text($item) . "\\\\\n";
            next;
        }
        
        # Handle alerts `> [!NOTE]`, `> [!CAUTION]`
        if ($line =~ /^>\s*\[!(NOTE|CAUTION|WARNING|IMPORTANT|TIP)\]\s*(.*)$/i) {
            my $type = ucfirst(lc($1));
            my $rest = $2;
            push @out, "\\textbf{Operational $type}: " . format_text($rest) . "\n";
            next;
        }
        if ($line =~ /^>\s*(.*)$/) {
            my $rest = $1;
            push @out, format_text($rest) . "\n";
            next;
        }
        
        # Normal prose line
        if ($line =~ /\S/) {
            push @out, format_text($line) . "\n";
        } else {
            push @out, "\n";
        }
    }
    
    flush_table() if $in_table;
    push @out, "\n\\bigskip\n";
    return join("", @out);
}

# Main execution
open my $out_fh, '>:encoding(UTF-8)', $out_file or die "Cannot write to $out_file: $!";

print $out_fh <<'HEADER';
% ==============================================================================
% Appendix C: Batch 05 — Gated Linear Attention & Hierarchically Gated Recurrent Networks (GLA & HGRN2)
% ==============================================================================
\subsection*{Batch 05: Gated Linear Attention \& Hierarchically Gated Recurrent Networks (GLA \& HGRN2)}
\label{sec:appendix_c_batch05_gla_hgrn}

Batch~05 formalizes the mathematical and microarchitectural transition from quadratic softmax attention $\mathcal{O}(N^2)$ to linear-time associative memory and state space recurrence $\mathcal{O}(N)$. The ten production operators analyzed within this monograph span four dominant linear attention paradigms:
\begin{enumerate}
    \item \textbf{Data-Dependent Gated Linear Attention (GLA)}\\
    (\texttt{gated\_linear\_attn}, \texttt{gla\_attention}, \texttt{google\_linear\_attention}):\\
    Replacing static kernel mappings with dynamic, input-conditioned exponential decay gates that prevent associative memory saturation across unbounded context windows.
    \item \textbf{Hybrid Sliding-Window Recurrence \& Delta Memories}\\
    (\texttt{goldfinch\_ahn}, \texttt{goldfinch\_attention}, \texttt{griffin\_recurrence}):\\
    Marrying high-frequency local attention (sliding-window multi-query) with low-frequency global memory (Widrow-Hoff delta updates and Real-Gated Linear Recurrence Units).
    \item \textbf{Hierarchically Gated Recurrent Networks (HGRN-2)}\\
    (\texttt{hgrn2\_attention}, \texttt{hgrn2\_gated\_deltanet}, \texttt{hgrn2\_gated\_layer}):\\
    Leveraging convex combination forgetting dynamics with strict $L_\infty$ state norm boundedness, causal 1D depthwise convolutions, and matrix delta memories.
    \item \textbf{Grouped-Query Attention Subspace Sharing}\\
    (\texttt{gqa\_attention}):\\
    Providing the foundational bandwidth-bound baseline that reduces autoregressive KV-cache footprint by an integer grouping ratio $G = H_Q / H_{KV}$ while retaining dense quadratic attention dynamics.
\end{enumerate}

\begin{table*}[htbp]
\centering
\caption{Empirical Telemetry and Operational Hardware Metrics for Batch 05: Gated Linear Attention \& Hierarchically Gated Recurrent Networks (GLA \& HGRN2). All benchmarks executed on NVIDIA GeForce RTX 4070 Laptop GPU (AD104, SM 8.9) under identical context parameters ($B=1, H=12, N=2048, d=64$).}
\label{tab:batch_05_summary}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.12}
\resizebox{\linewidth}{!}{%
\begin{tabular}{l l c c c c c c}
\toprule
\textbf{Module Index \& Key} & \textbf{Architectural Family} & \textbf{Backend} & \textbf{Latency} & \textbf{Speedup} & \textbf{Peak VRAM} & \textbf{SST-2 Acc} & \textbf{Gate Status} \\
\midrule
Module 042: \texttt{gated\_linear\_attn} & Gated Linear Attention & Level 2 & $4.396\text{ ms}$ & $1.00\times$ & $346.6\text{ MB}$ & $88.90\%$ & \passfast{} \\
Module 043: \texttt{gla\_attention} & Gated Linear Attention & Level 2 & $3.228\text{ ms}$ & $1.00\times$ & $198.5\text{ MB}$ & $90.93\%$ & \passfast{} \\
Module 044: \texttt{goldfinch\_ahn} & SSM Hybrid (AHN) & Level 1 & $4.722\text{ ms}$ & $1.00\times$ & $348.7\text{ MB}$ & $94.27\%$ & \passfast{} \\
Module 045: \texttt{goldfinch\_attention} & Block-Partitioned Delta & Level 1 & $11.660\text{ ms}$ & $1.00\times$ & $336.0\text{ MB}$ & $91.06\%$ & \passfast{} \\
Module 046: \texttt{google\_linear\_attention} & Linear Kernelized Attn & Level 1 & $1.964\text{ ms}$ & $1.00\times$ & $289.4\text{ MB}$ & $92.38\%$ & \passfast{} \\
Module 047: \texttt{gqa\_attention} & Grouped-Query Attention & Level 1 & $0.906\text{ ms}$ & $1.00\times$ & $630.8\text{ MB}$ & $95.00\%$ & \passfast{} \\
Module 048: \texttt{griffin\_recurrence} & Real-Gated Recurrence & Level 2 & $1.953\text{ ms}$ & $1.00\times$ & $344.4\text{ MB}$ & $91.97\%$ & \passfast{} \\
Module 049: \texttt{hgrn2\_attention} & Convex Gated RNN & Level 1 & $1.808\text{ ms}$ & $1.00\times$ & $445.1\text{ MB}$ & $94.15\%$ & \passfast{} \\
Module 050: \texttt{hgrn2\_gated\_deltanet} & Matrix Delta Memory & Level 2 & $7.653\text{ ms}$ & $1.00\times$ & $283.1\text{ MB}$ & $92.00\%$ & \passfast{} \\
Module 051: \texttt{hgrn2\_gated\_layer} & Complete Modular Block & Level 1 & $1.808\text{ ms}$ & $1.00\times$ & $385.2\text{ MB}$ & $93.00\%$ & \passfast{} \\
\bottomrule
\end{tabular}%
}
\end{table*}

\bigskip

HEADER

for my $mod (@modules) {
    my ($idx, $key, $title, $path) = @$mod;
    print "Processing Module $idx: $key...\n";
    my $tex = process_spec_file($idx, $key, $title, $path);
    print $out_fh $tex;
}

close $out_fh;
print "Successfully generated $out_file\n";
