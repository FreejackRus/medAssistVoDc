import { useNavigate } from "react-router";
import { BookOpen, Download, BrainCircuit, Loader2, ArrowRight } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/toast";
import { useDocuments, useImportRecommendation } from "@/hooks/useDocuments";
import type { ClinicalRecommendation } from "@/hooks/useClinicalRecs";

interface Props {
  rec: ClinicalRecommendation;
}

const CR_API = "https://apicr.minzdrav.gov.ru/api.ashx";

export default function RecommendationCard({ rec }: Props) {
  const navigate = useNavigate();
  const { data: docs } = useDocuments();
  const importMutation = useImportRecommendation();
  const { toast } = useToast();

  const expectedFilename = `${rec.title}.pdf`;
  const alreadyImported = docs?.some((d) => d.filename === expectedFilename && d.status === "ready");

  const pdfUrl = rec.code_version
    ? `${CR_API}?id=${rec.code_version}&op=GetClinrecPdf`
    : null;

  const handleAnalyze = () => {
    if (!rec.code_version) return;
    importMutation.mutate(
      { code_version: rec.code_version, title: rec.title },
      {
        onSuccess: (doc) => {
          if (doc.status === "ready") {
            toast("Рекомендация загружена и обработана", "success");
            navigate("/");
          } else {
            toast("Не удалось обработать рекомендацию", "error");
          }
        },
        onError: (err) => toast(err.message, "error"),
      },
    );
  };

  return (
    <Card className="transition-colors hover:bg-accent/30">
      <CardContent className="flex items-start gap-3 p-4">
        <BookOpen className="mt-0.5 h-5 w-5 shrink-0 text-muted-foreground" />
        <div className="min-w-0 flex-1 space-y-1.5">
          <p className="text-sm font-medium leading-tight">{rec.title}</p>
          {rec.annotation && (
            <p className="text-xs text-muted-foreground line-clamp-2">
              {rec.annotation}
            </p>
          )}
          <div className="flex flex-wrap items-center gap-1.5">
            {rec.mkb_code && (
              <Badge variant="secondary" className="text-xs">
                {rec.mkb_code}
              </Badge>
            )}
            {rec.age_group && (
              <Badge variant="secondary" className="text-xs">
                {rec.age_group}
              </Badge>
            )}
            {rec.publishdate && (
              <span className="text-xs text-muted-foreground">
                {rec.publishdate}
              </span>
            )}
          </div>
        </div>
        <div className="flex shrink-0 self-center gap-2">
          {rec.code_version && (
            alreadyImported ? (
              <Button
                size="sm"
                variant="outline"
                onClick={() => navigate("/")}
                className="gap-2"
              >
                <ArrowRight className="h-4 w-4" />
                Перейти
              </Button>
            ) : (
              <Button
                size="sm"
                onClick={handleAnalyze}
                disabled={importMutation.isPending}
                title="Скачать PDF с Минздрава, обработать и создать документ для анализа"
                className="gap-2"
              >
                {importMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <BrainCircuit className="h-4 w-4" />
                )}
                {importMutation.isPending ? "Загрузка..." : "Анализировать"}
              </Button>
            )
          )}
          {pdfUrl && (
            <a href={pdfUrl} target="_blank" rel="noopener noreferrer" download>
              <Button variant="outline" size="sm" className="gap-2">
                <Download className="h-4 w-4" />
                PDF
              </Button>
            </a>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
