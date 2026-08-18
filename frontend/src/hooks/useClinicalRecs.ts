import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";

export interface ClinicalRecommendation {
  id: string;
  title: string;
  description: string;
  annotation: string;
  publishdate: string;
  status: number;
  organization: string;
  mkb_code: string;
  category: string;
  age_group: string;
  keywords: string;
  version: string;
  code_version: string;
}

interface RecsResponse {
  success: boolean;
  total: number;
  recommendations: ClinicalRecommendation[];
}

export function useClinicalRecs() {
  return useQuery<ClinicalRecommendation[]>({
    queryKey: ["clinical-recommendations"],
    queryFn: async () => {
      const data = await apiFetch<RecsResponse>("/clinical-recommendations");
      return data.recommendations;
    },
    staleTime: 5 * 60 * 1000,
  });
}
