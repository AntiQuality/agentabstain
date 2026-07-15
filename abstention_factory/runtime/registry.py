from __future__ import annotations


from abstention_factory.environments.agriculture_and_yield.environment import AgricultureAndYieldEnvironment
from abstention_factory.environments.astrology_charting.environment import AstrologyChartingEnvironment
from abstention_factory.environments.autonomous_delivery_and_routing.environment import AutonomousDeliveryAndRoutingEnvironment
from abstention_factory.environments.calendar.environment import CalendarEnvironment
from abstention_factory.environments.chat_and_group_management.environment import ChatAndGroupManagementEnvironment
from abstention_factory.environments.clinical_records_and_claims.environment import ClinicalRecordsAndClaimsEnvironment
from abstention_factory.environments.compliance_and_contracts.environment import ComplianceAndContractsEnvironment
from abstention_factory.environments.consumer_banking.environment import ConsumerBankingEnvironment
from abstention_factory.environments.consumer_health_portal.environment import ConsumerHealthPortalEnvironment
from abstention_factory.environments.crm_and_company_lookup.environment import CrmAndCompanyLookupEnvironment
from abstention_factory.environments.device_privacy_and_focus.environment import DevicePrivacyAndFocusEnvironment
from abstention_factory.environments.disaster_relief_operations.environment import DisasterReliefOperationsEnvironment
from abstention_factory.environments.document_authoring_and_publication.environment import DocumentAuthoringAndPublicationEnvironment
from abstention_factory.environments.documents_and_analysis.environment import DocumentsAndAnalysisEnvironment
from abstention_factory.environments.education_and_campus_portals.environment import EducationAndCampusPortalsEnvironment
from abstention_factory.environments.filesystem.environment import FilesystemEnvironment
from abstention_factory.environments.fitness_and_wellness_logs.environment import FitnessAndWellnessLogsEnvironment
from abstention_factory.environments.flight_and_travel_management.environment import FlightAndTravelManagementEnvironment
from abstention_factory.environments.gmail_and_email_records.environment import GmailAndEmailRecordsEnvironment
from abstention_factory.environments.home_medication_inventory.environment import HomeMedicationInventoryEnvironment
from abstention_factory.environments.identity_credit_and_collections.environment import IdentityCreditAndCollectionsEnvironment
from abstention_factory.environments.industrial_and_infrastructure_control.environment import IndustrialAndInfrastructureControlEnvironment
from abstention_factory.environments.maps_and_navigation.environment import MapsAndNavigationEnvironment
from abstention_factory.environments.metrics_and_spreadsheet_analysis.environment import MetricsAndSpreadsheetAnalysisEnvironment
from abstention_factory.environments.notes_and_reference.environment import NotesAndReferenceEnvironment
from abstention_factory.environments.personal_profile_and_contacts.environment import PersonalProfileAndContactsEnvironment
from abstention_factory.environments.phone_and_messages.environment import PhoneAndMessagesEnvironment
from abstention_factory.environments.project_logs.environment import ProjectLogsEnvironment
from abstention_factory.environments.retail_orders.environment import RetailOrdersEnvironment
from abstention_factory.environments.science_and_environment_data.environment import ScienceAndEnvironmentDataEnvironment
from abstention_factory.environments.security_and_privacy_admin.environment import SecurityAndPrivacyAdminEnvironment
from abstention_factory.environments.smart_home.environment import SmartHomeEnvironment
from abstention_factory.environments.social_media_dataset_analysis.environment import SocialMediaDatasetAnalysisEnvironment
from abstention_factory.environments.spotify.environment import SpotifyEnvironment
from abstention_factory.environments.store_procurement_and_inventory.environment import StoreProcurementAndInventoryEnvironment
from abstention_factory.environments.system_operations.environment import SystemOperationsEnvironment
from abstention_factory.environments.trading_and_portfolio.environment import TradingAndPortfolioEnvironment
from abstention_factory.environments.vehicle_status_and_control.environment import VehicleStatusAndControlEnvironment
from abstention_factory.environments.venmo_and_shared_expenses.environment import VenmoAndSharedExpensesEnvironment
from abstention_factory.environments.weather_and_alerts.environment import WeatherAndAlertsEnvironment
from abstention_factory.environments.web_and_cms.environment import WebAndCmsEnvironment
from abstention_factory.environments.workforce_and_hr.environment import WorkforceAndHrEnvironment


ENVIRONMENT_REGISTRY = {
    "agriculture_and_yield": AgricultureAndYieldEnvironment,
    "astrology_charting": AstrologyChartingEnvironment,
    "autonomous_delivery_and_routing": AutonomousDeliveryAndRoutingEnvironment,
    "calendar": CalendarEnvironment,
    "chat_and_group_management": ChatAndGroupManagementEnvironment,
    "clinical_records_and_claims": ClinicalRecordsAndClaimsEnvironment,
    "compliance_and_contracts": ComplianceAndContractsEnvironment,
    "consumer_banking": ConsumerBankingEnvironment,
    "consumer_health_portal": ConsumerHealthPortalEnvironment,
    "crm_and_company_lookup": CrmAndCompanyLookupEnvironment,
    "device_privacy_and_focus": DevicePrivacyAndFocusEnvironment,
    "disaster_relief_operations": DisasterReliefOperationsEnvironment,
    "document_authoring_and_publication": DocumentAuthoringAndPublicationEnvironment,
    "documents_and_analysis": DocumentsAndAnalysisEnvironment,
    "education_and_campus_portals": EducationAndCampusPortalsEnvironment,
    "filesystem": FilesystemEnvironment,
    "fitness_and_wellness_logs": FitnessAndWellnessLogsEnvironment,
    "flight_and_travel_management": FlightAndTravelManagementEnvironment,
    "gmail_and_email_records": GmailAndEmailRecordsEnvironment,
    "home_medication_inventory": HomeMedicationInventoryEnvironment,
    "identity_credit_and_collections": IdentityCreditAndCollectionsEnvironment,
    "industrial_and_infrastructure_control": IndustrialAndInfrastructureControlEnvironment,
    "maps_and_navigation": MapsAndNavigationEnvironment,
    "metrics_and_spreadsheet_analysis": MetricsAndSpreadsheetAnalysisEnvironment,
    "notes_and_reference": NotesAndReferenceEnvironment,
    "personal_profile_and_contacts": PersonalProfileAndContactsEnvironment,
    "phone_and_messages": PhoneAndMessagesEnvironment,
    "project_logs": ProjectLogsEnvironment,
    "retail_orders": RetailOrdersEnvironment,
    "science_and_environment_data": ScienceAndEnvironmentDataEnvironment,
    "security_and_privacy_admin": SecurityAndPrivacyAdminEnvironment,
    "smart_home": SmartHomeEnvironment,
    "social_media_dataset_analysis": SocialMediaDatasetAnalysisEnvironment,
    "spotify": SpotifyEnvironment,
    "store_procurement_and_inventory": StoreProcurementAndInventoryEnvironment,
    "system_operations": SystemOperationsEnvironment,
    "trading_and_portfolio": TradingAndPortfolioEnvironment,
    "vehicle_status_and_control": VehicleStatusAndControlEnvironment,
    "venmo_and_shared_expenses": VenmoAndSharedExpensesEnvironment,
    "weather_and_alerts": WeatherAndAlertsEnvironment,
    "web_and_cms": WebAndCmsEnvironment,
    "workforce_and_hr": WorkforceAndHrEnvironment,
}


def get_environment_class(env_type: str):
    if env_type not in ENVIRONMENT_REGISTRY:
        raise ValueError(
            f"Unknown environment '{env_type}'. Available: {list(ENVIRONMENT_REGISTRY.keys())}"
        )
    return ENVIRONMENT_REGISTRY[env_type]


def list_environments() -> list[str]:
    return list(ENVIRONMENT_REGISTRY)


def build_multi_environment(env_types, initial_states):
    """Factory for MultiEnvironment. Imported lazily to avoid a cycle during
    registry construction (multi.py imports from this module)."""
    from abstention_factory.runtime.multi import build_multi_environment as _build
    return _build(env_types, initial_states)
